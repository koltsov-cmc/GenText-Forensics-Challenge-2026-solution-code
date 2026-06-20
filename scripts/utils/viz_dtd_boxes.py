#!/usr/bin/env python
"""Visualise DTD-sourced bounding boxes on source images.

Reads augmentation log entries from JSON files, extracts DTD-produced bboxes,
and draws numbered red rectangles on the corresponding source images.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


_BBOX_COLOR = "#ff2e2e"
_DPI = 150


# =========================================================================== #
# Discovery
# =========================================================================== #
def discover_json(json_dir: Path, parts, limit: int, pattern: str = "*.json") -> list[Path]:
    files: list[Path] = []
    if parts:
        for part in parts:
            pd = json_dir / part
            if not pd.is_dir():
                print(f"  [warn] part dir missing: {pd}")
                continue
            files.extend(sorted(pd.glob(pattern)))
    else:
        files = sorted(json_dir.rglob(pattern))
    if limit > 0:
        files = files[:limit]
    return files


def find_image(image_dir: Path, stem: str, image_name: str = "") -> Path | None:
    """Locate source image for a given stem."""
    if image_name:
        cand = image_dir / image_name
        if cand.exists():
            return cand
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        cand = image_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    # Recursive search inside partXXX subfolders
    for ext in (".jpg", ".jpeg", ".png"):
        matches = list(image_dir.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


# =========================================================================== #
# Extraction
# =========================================================================== #
def extract_dtd_boxes(payload: dict) -> list[list[int]]:
    """Return list of 'new' bboxes for entries with source == 'dtd'."""
    log = (payload.get("augmentation") or {}).get("log_ocr_dtd_only", [])
    boxes = []
    for entry in log:
        if entry.get("source") == "dtd" and entry.get("kept") is True:
            bb = entry.get("new")
            if bb and len(bb) == 4:
                boxes.append([int(v) for v in bb])
    return boxes


# =========================================================================== #
# Drawing
# =========================================================================== #
def draw_boxes(image_pil: Image.Image, boxes: list[list[int]], out_path: Path) -> None:
    """Draw numbered red bounding boxes on the image, saved to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    img_arr = np.asarray(image_pil.convert("RGB"))
    H, W = img_arr.shape[:2]

    fig_w_in = W / _DPI
    fig_h_in = H / _DPI
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=_DPI)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(image_pil, extent=[0, W, H, 0])

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        rect = mpatches.Rectangle(
            (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
            linewidth=2.5, edgecolor=_BBOX_COLOR, facecolor="none",
        )
        ax.add_patch(rect)

        bbox_h = max(1, y2 - y1)
        desired_text_px = 1.25 * bbox_h
        fontsize = max(8.0, desired_text_px * 72 / _DPI * 0.75)
        ax.text(
            x1, max(0, y1 - 4), str(i),
            color="white", fontsize=fontsize, fontweight="bold",
            bbox=dict(facecolor=_BBOX_COLOR, edgecolor="none",
                      alpha=0.9, pad=1),
            zorder=5,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, facecolor="white", bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)


# =========================================================================== #
# Per-file processing
# =========================================================================== #
def process_one_json(
    json_path: Path,
    *,
    json_dir: Path,
    image_dir: Path,
    out_dir: Path,
) -> str:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [error] cannot read {json_path.name}: {exc!r}", flush=True)
        return "error"

    stem = payload.get("stem") or json_path.stem.replace("_dtd_ocr", "").replace("_ocr", "")
    boxes = extract_dtd_boxes(payload)
    if not boxes:
        print(f"  [skip] {stem}: no DTD boxes in log_ocr_dtd_only", flush=True)
        return "no_dtd"

    img_path = find_image(image_dir, stem, payload.get("image_name", ""))
    if img_path is None:
        print(f"  [skip] {stem}: source image not found", flush=True)
        return "no_image"

    try:
        image_pil = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"  [skip] {stem}: failed to open image: {exc!r}", flush=True)
        return "bad_image"

    try:
        rel = json_path.parent.relative_to(json_dir)
    except ValueError:
        rel = Path("")
    out_png = out_dir / rel / f"{stem}_dtd_boxes.png"
    draw_boxes(image_pil, boxes, out_png)
    print(f"  [ok] {stem}: {len(boxes)} DTD box(es) -> {out_png.name}", flush=True)
    return "ok"


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json_dir", required=True,
                    help="Directory with augmented JSON files.")
    ap.add_argument("--image_dir", required=True,
                    help="Directory with source document images.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for visualisations.")
    ap.add_argument("--parts", nargs="*", default=None,
                    help="Optional partXXX subdirs to process.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N JSONs (0 = all).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    json_dir = Path(args.json_dir).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not json_dir.is_dir():
        raise SystemExit(f"--json_dir is not a directory: {json_dir}")
    if not image_dir.is_dir():
        raise SystemExit(f"--image_dir is not a directory: {image_dir}")

    files = discover_json(json_dir, args.parts, args.limit)
    if not files:
        print("[done] no JSON files found")
        return 0
    print(f"[run] {len(files)} JSON file(s)")

    counters = {"ok": 0, "no_dtd": 0, "no_image": 0, "bad_image": 0, "error": 0}
    for i, jp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {jp.name}", flush=True)
        status = process_one_json(jp, json_dir=json_dir, image_dir=image_dir, out_dir=out_dir)
        counters[status] = counters.get(status, 0) + 1

    print(f"\n[done] {counters}")
    print(f"[done] outputs under {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
