#!/usr/bin/env python
"""Run PaddleOCR over a directory of images and save {stem}_ocr.json.

Performs language detection + OCR on each image, saving compact OCR JSON
(preserving partXXX subdirectory structure). Supports multi-GPU sharding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Path setup (same as extract_dtd_ocr_heatmaps.py)
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = Path(__file__).resolve().parent
_TOOLKIT_ROOT = None
for r in (_SCRIPT_DIR, _SCRIPT_DIR.parent, _SCRIPT_DIR.parent.parent):
    if (r / "realtext_v2").is_dir() or (r / "ForensicHub").is_dir():
        _TOOLKIT_ROOT = r
        break
if _TOOLKIT_ROOT is None:
    _TOOLKIT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_TOOLKIT_ROOT))

# OCR import (same engine class as extract_dtd_ocr_heatmaps.py)
sys.path.insert(0, str(_TOOLKIT_ROOT / "scripts"))
from run_paddle_sobel import run_paddle_ocr_with_lang_detect  # noqa: E402


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- #
# OCR wrapper
# --------------------------------------------------------------------------- #
def _run_ocr(
    image_path: Path,
    candidate_langs: list[str],
    *,
    ocr_engine: any,
    gpu: bool,
) -> dict:
    result = run_paddle_ocr_with_lang_detect(
        image_path, candidate_langs=candidate_langs, gpu=gpu,
        mag_ratio=1.0, verbose=False, engine=ocr_engine,
    )
    result["selected_language"] = result["lang"]
    return result


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #
def process_one(
    image_path: Path,
    *,
    args,
    image_dir: Path,
    ocr_engine: any,
    out_json_dir: Path,
) -> str:
    """Returns status string ('ok' / 'skip_exists' / 'error')."""
    import torch

    stem = image_path.stem

    # Preserve subdirectory structure (e.g. part000/...)
    rel_dir = image_path.parent.relative_to(image_dir)
    if rel_dir == Path("."):
        rel_dir = Path("")

    json_dir = out_json_dir / rel_dir
    json_path = json_dir / f"{stem}_ocr.json"

    if args.skip_existing and json_path.exists():
        return "skip_exists"

    # ---- Open image to get size ----
    try:
        image_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image_pil.size
    except Exception as exc:
        print(f"  [skip] cannot open image {image_path.name}: {exc!r}", flush=True)
        return "error"

    # ---- OCR ----
    ocr_ok = True
    try:
        ocr_result = _run_ocr(
            image_path,
            candidate_langs=[s.strip() for s in args.langs.split(",")
                             if s.strip()],
            ocr_engine=ocr_engine,
            gpu=torch.cuda.is_available(),
        )
    except Exception as exc:
        print(f"  [warn] OCR failed for {image_path.name}: {exc!r}", flush=True)
        ocr_ok = False

    if ocr_ok:
        ocr_items = ocr_result.get("ocr_items", []) or []
        reading_order = ocr_result.get("reading_order_text", "") or ""
        ocr_lang = ocr_result.get("lang", "unknown")
        n_items = ocr_result.get("n_items", len(ocr_items))
    else:
        ocr_items = []
        reading_order = ""
        ocr_lang = "unknown"
        n_items = 0

    # ---- ocr_input payload (matches the schema used by prerender_prompts.py) ----
    ocr_input = {
        "lang":               ocr_lang,
        "n_items":            n_items,
        "ocr_items":          [
            {
                "id":         it.get("id"),
                "text":       it.get("text", ""),
                "bbox":       it.get("bbox"),
                "confidence": round(float(it.get("confidence", 0.0)), 3),
            } for it in ocr_items
        ],
        "reading_order_text": reading_order,
    }

    # ---- Write JSON ----
    payload = {
        "image_name":   image_path.name,
        "stem":         stem,
        "image_size":   [orig_w, orig_h],
        "ocr_n_items":  n_items,
        "ocr_lang":     ocr_lang,
        "ocr_input":    ocr_input,
    }
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"  [ok] {image_path.name}  {n_items} OCR items ({ocr_lang})")
    return "ok"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image_dir", required=True,
                    help="Directory of source images.")
    ap.add_argument("--out_json_dir", required=True,
                    help="Output dir for {stem}_ocr.json files.")

    # OCR
    ap.add_argument("--langs", default="en,ch,th,ms,id,ar",
                    help="Comma-separated PaddleOCR language codes for "
                         "language detection.")

    # Runtime
    ap.add_argument("--device", default="cuda",
                    help="Used only in single-process mode (--num_gpus 1).")
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="Number of GPUs to shard work across (default 8). "
                         "Files are split modulo num_gpus by sorted order; "
                         "each GPU runs one worker process. Set to 1 to run "
                         "in-process with no fork.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process only the first N images "
                         "(applied BEFORE sharding).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip images whose JSON already exists.")
    ap.add_argument("--skip_png", action="store_true",
                    help="Skip all .png files (do not run OCR on them).")

    # Internal — used by spawned workers, not by users.
    ap.add_argument("--_worker_gpu_id", type=int, default=-1,
                    help=argparse.SUPPRESS)
    return ap.parse_args()


def gather_images(image_dir: Path, limit: int, skip_png: bool = False) -> list[Path]:
    paths = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _IMG_EXTS
        and not (skip_png and p.suffix.lower() == ".png")
    )
    if limit > 0:
        paths = paths[:limit]
    return paths


def _run_worker(
    gpu_id: int,
    image_paths: list[Path],
    args,
    return_queue=None,
) -> dict:
    """Worker entrypoint. Pins to `gpu_id` (must be called BEFORE any torch
    CUDA import). Processes the given shard of image paths.

    Returns a counters dict {"ok": .., "skip_exists": .., "error": ..}.
    If `return_queue` is provided (multi-process mode), the dict is also
    placed on it.
    """
    import os
    # Pin to single visible GPU — must happen before torch.cuda init.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Now safe to import torch
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[worker gpu={gpu_id}] start  shard_size={len(image_paths)}  "
          f"device={device}  visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)

    image_dir    = Path(args.image_dir).expanduser().resolve()
    out_json_dir = Path(args.out_json_dir).expanduser().resolve()

    # Reusable OCR engine (one per worker, not per image)
    t0 = time.time()
    from run_paddle_sobel import _PaddleOcrEngine
    ocr_engine = _PaddleOcrEngine(
        gpu=torch.cuda.is_available(), gpu_id=0,
    )
    print(f"[worker gpu={gpu_id}] ocr engine ready in {time.time()-t0:.1f}s",
          flush=True)

    counters = {"ok": 0, "skip_exists": 0, "error": 0}
    try:
        for i, image_path in enumerate(image_paths, start=1):
            print(f"[worker gpu={gpu_id}] [{i}/{len(image_paths)}] "
                  f"{image_path.name}", flush=True)
            try:
                status = process_one(
                    image_path, args=args, image_dir=image_dir,
                    ocr_engine=ocr_engine,
                    out_json_dir=out_json_dir,
                )
            except Exception as exc:
                print(f"  [worker gpu={gpu_id}] [error] {exc!r}", flush=True)
                import traceback
                traceback.print_exc()
                status = "error"
            counters[status] = counters.get(status, 0) + 1
    finally:
        print(f"[worker gpu={gpu_id}] done  {counters}", flush=True)
        if return_queue is not None:
            return_queue.put({"gpu_id": gpu_id, "counters": counters})
    return counters


def _shard_paths(all_paths: list[Path], n_shards: int) -> list[list[Path]]:
    """Split `all_paths` into `n_shards` contiguous shards as evenly as
    possible."""
    n = len(all_paths)
    base, extra = divmod(n, n_shards)
    shards = []
    idx = 0
    for k in range(n_shards):
        size = base + (1 if k < extra else 0)
        shards.append(all_paths[idx: idx + size])
        idx += size
    return shards


def main() -> int:
    args = parse_args()

    # ============================================================ #
    # Internal worker invocation path
    # ============================================================ #
    if args._worker_gpu_id >= 0:
        all_paths = gather_images(
            Path(args.image_dir).expanduser().resolve(), args.limit,
            skip_png=args.skip_png,
        )
        _run_worker(args._worker_gpu_id, all_paths, args)
        return 0

    # ============================================================ #
    # Parent / single-process entrypoint
    # ============================================================ #
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"--image_dir not a directory: {image_dir}")
    out_json_dir = Path(args.out_json_dir).expanduser().resolve()
    out_json_dir.mkdir(parents=True, exist_ok=True)

    image_paths = gather_images(image_dir, args.limit, skip_png=args.skip_png)
    if not image_paths:
        print("[error] no images found")
        return 1
    if args.skip_png:
        print("[run] skipping .png files (--skip_png)")
    print(f"[run] {len(image_paths)} image(s) from {image_dir}")
    print(f"[out] json -> {out_json_dir}")

    n_gpus = max(1, args.num_gpus)
    if n_gpus > len(image_paths):
        n_gpus = len(image_paths)
        print(f"[run] clamping --num_gpus to {n_gpus} (== num images)")

    # ------------------------------------------------------------ #
    # Single-process mode
    # ------------------------------------------------------------ #
    if n_gpus <= 1:
        import os
        if args.device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
        gpu_id = 0
        counters = _run_worker(gpu_id, image_paths, args)
        print(f"\n[done] {counters}")
        return 0

    # ------------------------------------------------------------ #
    # Multi-GPU mode: spawn one worker process per GPU
    # ------------------------------------------------------------ #
    shards = _shard_paths(image_paths, n_gpus)
    for k, shard in enumerate(shards):
        print(f"[shard] gpu={k}  size={len(shard)}  "
              f"first={shard[0].name if shard else '-'}  "
              f"last={shard[-1].name if shard else '-'}")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    return_queue = ctx.Queue()
    procs: list = []
    t_start = time.time()
    for gpu_id, shard in enumerate(shards):
        if not shard:
            continue
        p = ctx.Process(
            target=_run_worker,
            args=(gpu_id, shard, args, return_queue),
            name=f"ocr-extractor-gpu{gpu_id}",
        )
        p.start()
        procs.append(p)
        print(f"[spawn] gpu={gpu_id}  pid={p.pid}  shard={len(shard)}")

    # Collect results
    agg = {"ok": 0, "skip_exists": 0, "error": 0}
    per_gpu_counters: dict[int, dict] = {}
    n_finished = 0
    finished_gpu_ids: set[int] = set()
    while n_finished < len(procs):
        try:
            msg = return_queue.get(timeout=10)
        except Exception:
            for p in procs:
                if p.pid is None:
                    continue
                if not p.is_alive() and p.pid not in finished_gpu_ids:
                    gpu_id = int(p.name.replace("ocr-extractor-gpu", ""))
                    print(f"[parent] gpu={gpu_id} process crashed "
                          f"(exitcode={p.exitcode})", flush=True)
                    per_gpu_counters[gpu_id] = {"crashed": True}
                    finished_gpu_ids.add(p.pid)
                    n_finished += 1
            continue
        gpu_id = msg["gpu_id"]
        counters = msg["counters"]
        per_gpu_counters[gpu_id] = counters
        for k, v in counters.items():
            agg[k] = agg.get(k, 0) + v
        n_finished += 1
        print(f"[parent] gpu={gpu_id} finished  {counters}  "
              f"({n_finished}/{len(procs)})")

    # Final join
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            print(f"[parent] WARN: process {p.name} (pid={p.pid}) still alive "
                  f"after queue drain; terminating")
            p.terminate()
            p.join(timeout=5)

    elapsed = time.time() - t_start
    print(f"\n[done] {len(procs)} worker(s)  elapsed={elapsed:.1f}s  "
          f"aggregated={agg}")
    print(f"[done] per-gpu: " + ", ".join(
        f"gpu{g}={c}" for g, c in sorted(per_gpu_counters.items())
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
