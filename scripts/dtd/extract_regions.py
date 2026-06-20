#!/usr/bin/env python
"""Extract DTD region bounding boxes at multiple probability thresholds.

Two modes: (A) from pre-computed .npy probability maps, or (B) run DTD
inference directly on images with optional TTA.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Optional DTD inference path (for TTA mode)
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

_DTD_SCRIPT_DIR = _TOOLKIT_ROOT / "ForensicHub" / "dtd_train"
sys.path.insert(0, str(_DTD_SCRIPT_DIR))
try:
    import run_doc_forensics_inference as _dtd  # noqa: E402
except Exception as _exc:  # pragma: no cover
    _dtd = None


# --------------------------------------------------------------------------- #
# Copied verbatim from extract_dtd_ocr_heatmaps.py
# --------------------------------------------------------------------------- #
def _largest_internal_gap(present: np.ndarray, lo: int, hi: int):
    best_len, best_split = 0, -1
    run_start = None
    for i in range(lo, hi + 1):
        if not present[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if run_start > lo and (i - 1) < hi:
                    run_len = i - run_start
                    if run_len > best_len:
                        best_len = run_len
                        best_split = (run_start + i) // 2
                run_start = None
    return best_len, best_split


def _boxes_from_submask(sub: np.ndarray, ox: int, oy: int, min_area: int,
                        min_gap: int = 2, fill_thresh: float = 0.45,
                        depth: int = 0, max_depth: int = 8) -> list[list[int]]:
    ys, xs = np.where(sub)
    if ys.size == 0:
        return []
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h, w = (y2 - y1 + 1), (x2 - x1 + 1)
    area = int(ys.size)
    fill = area / float(max(1, h * w))

    if depth >= max_depth or fill >= fill_thresh or area < 2 * min_area:
        if area >= min_area:
            return [[ox + x1, oy + y1, ox + x2 + 1, oy + y2 + 1]]
        return []

    row_present = sub.any(axis=1)
    col_present = sub.any(axis=0)
    rgap_len, rsplit = _largest_internal_gap(row_present, y1, y2)
    cgap_len, csplit = _largest_internal_gap(col_present, x1, x2)

    if max(rgap_len, cgap_len) < min_gap:
        if area >= min_area:
            return [[ox + x1, oy + y1, ox + x2 + 1, oy + y2 + 1]]
        return []

    out: list[list[int]] = []
    if rgap_len >= cgap_len:
        top = sub.copy(); top[rsplit:, :] = False
        bot = sub.copy(); bot[:rsplit, :] = False
        out += _boxes_from_submask(top, ox, oy, min_area, min_gap,
                                   fill_thresh, depth + 1, max_depth)
        out += _boxes_from_submask(bot, ox, oy, min_area, min_gap,
                                   fill_thresh, depth + 1, max_depth)
    else:
        left  = sub.copy(); left[:, csplit:]  = False
        right = sub.copy(); right[:, :csplit] = False
        out += _boxes_from_submask(left,  ox, oy, min_area, min_gap,
                                   fill_thresh, depth + 1, max_depth)
        out += _boxes_from_submask(right, ox, oy, min_area, min_gap,
                                   fill_thresh, depth + 1, max_depth)
    return out


def _label_components(mask: np.ndarray, connectivity: int):
    try:
        import cv2
        n, labels = cv2.connectedComponents(mask.astype(np.uint8),
                                            connectivity=connectivity)
        return n, labels
    except Exception:
        from scipy import ndimage
        if connectivity == 4:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:
            structure = np.ones((3, 3), dtype=int)
        labels, n = ndimage.label(mask, structure=structure)
        return n + 1, labels


def _morph_open(mask: np.ndarray, ksize: int) -> np.ndarray:
    if not ksize or ksize < 2:
        return mask
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    except Exception:
        from scipy import ndimage
        st = np.ones((ksize, ksize), dtype=bool)
        return ndimage.binary_opening(mask.astype(bool),
                                      structure=st).astype(np.uint8)


def _dtd_regions_from_prob(prob: np.ndarray, threshold: float = 0.98,
                            min_area: int = 200, *,
                            connectivity: int = 4,
                            morph_open_ksize: int = 3,
                            split_low_fill: bool = True,
                            fill_split_thresh: float = 0.45,
                            min_split_gap: int = 2) -> list[list[int]]:
    mask = (prob >= threshold).astype(np.uint8)
    mask = _morph_open(mask, morph_open_ksize)

    n, labels = _label_components(mask, connectivity)
    boxes: list[list[int]] = []
    for lab in range(1, n):
        comp = (labels == lab)
        if int(comp.sum()) < min_area:
            continue
        ys, xs = np.where(comp)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        if split_low_fill:
            sub = comp[y1:y2 + 1, x1:x2 + 1]
            boxes.extend(_boxes_from_submask(
                sub, ox=x1, oy=y1, min_area=min_area,
                min_gap=min_split_gap, fill_thresh=fill_split_thresh))
        else:
            boxes.append([x1, y1, x2 + 1, y2 + 1])

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Original prob-map mode
    ap.add_argument("--prob_dir", default=None,
                    help="Root with partXXX/{stem}_dtd_prob.npy files.")
    ap.add_argument("--out_dir", required=True,
                    help="Root output dir (will contain thresh_*/part*/).")
    ap.add_argument("--thresholds", default="0.2,0.3,0.45,0.5,0.6,0.7",
                    help="Comma-separated probability thresholds.")
    ap.add_argument("--min_area", type=int, default=200)
    ap.add_argument("--connectivity", type=int, default=4, choices=(4, 8))
    ap.add_argument("--morph_open_ksize", type=int, default=3)
    ap.add_argument("--fill_split_thresh", type=float, default=0.45)
    ap.add_argument("--min_split_gap", type=int, default=2)

    # New: image-based DTD inference with optional TTA
    ap.add_argument("--image_dir", default=None,
                    help="If set, run DTD inference on source images. Either "
                         "--prob_dir or --image_dir (or both) must be given.")
    ap.add_argument("--dtd_config", default=None,
                    help="DTD YAML config (required when --image_dir is used).")
    ap.add_argument("--dtd_checkpoint", default=None,
                    help="DTD .pth checkpoint (required when --image_dir is used).")
    ap.add_argument("--tta", action="store_true",
                    help="Enable test-time augmentation during DTD inference.")
    ap.add_argument("--tta_combine", default="min",
                    choices=("min", "mean", "median"),
                    help="How to combine per-pass probability maps: "
                         "min=conservative, mean=smoothing, median=middle.")
    ap.add_argument("--tta_passes", type=int, default=4, choices=(3, 4),
                    help="Number of TTA passes.")
    ap.add_argument("--jpeg_quality", type=int, default=95,
                    help="JPEG quality for DTD inference.")
    ap.add_argument("--save_prob", default=None,
                    help="If set, save generated prob maps to this dir "
                         "(mirrors partXXX layout).")
    ap.add_argument("--device", default="cuda",
                    help="Torch device for DTD inference.")

    # Filtering
    ap.add_argument("--skip_png", action="store_true",
                    help="Skip .png files when processing --image_dir.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process only first N images / prob maps.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    print(f"[config] thresholds={thresholds}")

    # ============================================================ #
    # Mode B: image-based DTD inference (optional TTA)
    # ============================================================ #
    if args.image_dir:
        if _dtd is None:
            raise SystemExit(
                "DTD module not available; cannot run --image_dir mode. "
                "Ensure ForensicHub is on PYTHONPATH."
            )
        if not args.dtd_config or not args.dtd_checkpoint:
            raise SystemExit(
                "--dtd_config and --dtd_checkpoint are required "
                "when --image_dir is used."
            )

        image_dir = Path(args.image_dir).expanduser().resolve()
        if not image_dir.is_dir():
            raise SystemExit(f"--image_dir not a directory: {image_dir}")

        import torch
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"[dtd] loading model on {device} ...")
        _dtd._setup_paths_and_registry()
        dtd_model, dtd_model_name, dtd_needs_dct = _dtd.build_model_and_load(
            args.dtd_config, args.dtd_checkpoint, device,
        )
        print(f"[dtd] loaded {dtd_model_name}")

        tta_offsets = _dtd._build_tta_offsets(args.tta, args.tta_passes)

        _IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        image_paths = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in _IMG_EXTS
            and not (args.skip_png and p.suffix.lower() == ".png")
        )
        if args.limit > 0:
            image_paths = image_paths[:args.limit]
        print(f"[scan] {len(image_paths)} image(s) from {image_dir}")
        if not image_paths:
            print("[abort] no images found")
            return 1

        for i, image_path in enumerate(image_paths, start=1):
            rel_dir = image_path.parent.relative_to(image_dir)
            if rel_dir == Path("."):
                rel_dir = Path("")
            stem = image_path.stem

            try:
                prob, image_pil = _dtd.infer_one_image(
                    image_path, dtd_model, dtd_model_name, dtd_needs_dct, device,
                    jpeg_quality=args.jpeg_quality,
                    tta_offsets=tta_offsets,
                    tta_combine=args.tta_combine,
                )
            except Exception as exc:
                print(f"  [skip] DTD failed for {image_path.name}: {exc!r}")
                continue

            if prob.ndim == 3:
                prob = prob.squeeze()

            # Optionally save prob map
            if args.save_prob:
                prob_dir = Path(args.save_prob).expanduser().resolve() / rel_dir
                prob_dir.mkdir(parents=True, exist_ok=True)
                np.save(prob_dir / f"{stem}_dtd_prob.npy",
                        prob.astype(np.float32))

            # Extract regions at every threshold
            for t in thresholds:
                regions = _dtd_regions_from_prob(
                    prob, threshold=t,
                    min_area=args.min_area,
                    connectivity=args.connectivity,
                    morph_open_ksize=args.morph_open_ksize,
                    split_low_fill=True,
                    fill_split_thresh=args.fill_split_thresh,
                    min_split_gap=args.min_split_gap,
                )
                t_dir = out_dir / f"thresh_{t:.2f}" / rel_dir
                t_dir.mkdir(parents=True, exist_ok=True)
                json_path = t_dir / f"{stem}_dtd_regions.json"
                json_path.write_text(
                    json.dumps({
                        "stem": stem,
                        "image_name": image_path.name,
                        "threshold": t,
                        "n_regions": len(regions),
                        "dtd_regions": regions,
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            if i % 500 == 0 or i == len(image_paths):
                print(f"  [{i}/{len(image_paths)}] {rel_dir}/{stem}  "
                      f"regions@0.2={len(_dtd_regions_from_prob(prob, threshold=0.2, min_area=args.min_area, connectivity=args.connectivity, morph_open_ksize=args.morph_open_ksize, split_low_fill=True, fill_split_thresh=args.fill_split_thresh, min_split_gap=args.min_split_gap))}")

        print(f"\n[done] outputs under {out_dir}/")
        for t in thresholds:
            t_dir = out_dir / f"thresh_{t:.2f}"
            n_json = len(list(t_dir.rglob("*_dtd_regions.json")))
            print(f"  thresh_{t:.2f}: {n_json} json file(s)")
        return 0

    # ============================================================ #
    # Mode A: read existing .npy prob maps (original behaviour)
    # ============================================================ #
    if not args.prob_dir:
        raise SystemExit("Either --prob_dir or --image_dir must be provided.")

    prob_dir = Path(args.prob_dir).expanduser().resolve()
    npy_files = sorted(p for p in prob_dir.rglob("*_dtd_prob.npy") if p.is_file())
    if args.limit > 0:
        npy_files = npy_files[:args.limit]
    print(f"[scan] {len(npy_files)} prob map(s) found under {prob_dir}")
    if not npy_files:
        print("[abort] no .npy files")
        return 1

    for i, npy_path in enumerate(npy_files, start=1):
        rel_dir = npy_path.parent.relative_to(prob_dir)
        stem = npy_path.name.replace("_dtd_prob.npy", "")

        prob = np.load(npy_path).astype(np.float32)
        if prob.ndim == 3:
            prob = prob.squeeze()

        for t in thresholds:
            regions = _dtd_regions_from_prob(
                prob, threshold=t,
                min_area=args.min_area,
                connectivity=args.connectivity,
                morph_open_ksize=args.morph_open_ksize,
                split_low_fill=True,
                fill_split_thresh=args.fill_split_thresh,
                min_split_gap=args.min_split_gap,
            )

            t_dir = out_dir / f"thresh_{t:.2f}" / rel_dir
            t_dir.mkdir(parents=True, exist_ok=True)
            json_path = t_dir / f"{stem}_dtd_regions.json"
            json_path.write_text(
                json.dumps({
                    "stem": stem,
                    "image_name": f"{stem}.jpg",
                    "threshold": t,
                    "n_regions": len(regions),
                    "dtd_regions": regions,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if i % 500 == 0 or i == len(npy_files):
            print(f"  [{i}/{len(npy_files)}] {rel_dir}/{stem}  "
                  f"regions@0.2={len(_dtd_regions_from_prob(prob, threshold=0.2, min_area=args.min_area, connectivity=args.connectivity, morph_open_ksize=args.morph_open_ksize, split_low_fill=True, fill_split_thresh=args.fill_split_thresh, min_split_gap=args.min_split_gap))}")

    print(f"\n[done] outputs under {out_dir}/")
    for t in thresholds:
        t_dir = out_dir / f"thresh_{t:.2f}"
        n_json = len(list(t_dir.rglob("*_dtd_regions.json")))
        print(f"  thresh_{t:.2f}: {n_json} json file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
