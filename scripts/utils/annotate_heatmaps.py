#!/usr/bin/env python
"""Draw numbered red DTD bounding boxes on existing heatmap PNGs.

Reads region data from {stem}_dtd_ocr.json (produced by extract_dtd_ocr.py)
and overlays numbered red rectangles on the corresponding heatmap images.
Bboxes are linearly rescaled if heatmap resolution differs from source.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402


_BBOX_COLOR = "#ff2e2e"
_TARGET_DPI = 150
_HM_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _find_heatmap_path(heatmap_dir: Path, stem: str) -> Optional[Path]:
    """Try common naming patterns. Flat layout, then nested partXXX/."""
    flat_cands = [
        f"{stem}.heatmap_annotated.png",  # in case caller already passed annotated dir
        f"{stem}.heatmap.png",
        f"{stem}_dtd.png",
        f"{stem}.dtd.png",
        f"{stem}_heatmap.png",
        f"{stem}.png",
    ]
    for name in flat_cands:
        p = heatmap_dir / name
        if p.exists():
            return p
    for child in heatmap_dir.iterdir():
        if child.is_dir():
            for name in flat_cands:
                p = child / name
                if p.exists():
                    return p
    return None


def annotate_one(
    dtd_ocr_path: Path,
    heatmap_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> str:
    print(1)
    try:
        rec = json.loads(dtd_ocr_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "decode_err"

    stem = rec.get("stem") or dtd_ocr_path.stem.replace("_dtd_ocr", "")
    dtd_regions = rec.get("dtd_regions") or []

    out_path = out_dir / f"{stem}.heatmap_annotated.png"
    if out_path.exists() and not overwrite:
        return "skip_exists"

    hm_path = _find_heatmap_path(heatmap_dir, stem)
    if hm_path is None:
        return "no_heatmap"

    heatmap = Image.open(hm_path).convert("RGB")
    hm_w, hm_h = heatmap.size

    # Auto-rescale bboxes if heatmap resolution differs from the source
    # image resolution recorded in the JSON.
    sx, sy = 1.0, 1.0
    image_size = rec.get("image_size")
    if image_size and len(image_size) == 2:
        src_w, src_h = int(image_size[0]), int(image_size[1])
        if src_w > 0 and src_h > 0 and (src_w, src_h) != (hm_w, hm_h):
            sx, sy = hm_w / src_w, hm_h / src_h

    # ---- Matplotlib figure: exact pixel dimensions, no cropping ----
    fig_w_in = hm_w / _TARGET_DPI
    fig_h_in = hm_h / _TARGET_DPI
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=_TARGET_DPI)
    ax.set_xlim(0, hm_w)
    ax.set_ylim(hm_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(heatmap, extent=[0, hm_w, hm_h, 0])

    for i, box in enumerate(dtd_regions, start=1):
        try:
            x1, y1, x2, y2 = (int(v) for v in box)
        except (TypeError, ValueError):
            continue

        if sx != 1.0 or sy != 1.0:
            x1 = int(round(x1 * sx))
            x2 = int(round(x2 * sx))
            y1 = int(round(y1 * sy))
            y2 = int(round(y2 * sy))

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        rect = mpatches.Rectangle(
            (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
            linewidth=2, edgecolor=_BBOX_COLOR, facecolor="none",
        )
        ax.add_patch(rect)

        bbox_h = max(1, y2 - y1)
        desired_text_px = 1.25 * bbox_h
        fontsize = max(6.0, desired_text_px * 72 / _TARGET_DPI * 0.75)
        ax.text(
            x1, max(0, y1 - 4), str(i),
            color="white", fontsize=fontsize, fontweight="bold",
            bbox=dict(facecolor=_BBOX_COLOR, edgecolor="none",
                       alpha=0.9, pad=1),
            zorder=5,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_TARGET_DPI, facecolor="white")
    plt.close(fig)
    return "ok"


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dtd_ocr_dir", required=True,
                    help="Dir of {stem}_dtd_ocr.json files.")
    ap.add_argument("--heatmap_dir", required=True,
                    help="Dir of raw heatmap PNGs (without bboxes).")
    ap.add_argument("--out_dir", required=True,
                    help="Output dir for annotated heatmaps.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dtd_ocr_dir = Path(args.dtd_ocr_dir).expanduser().resolve()
    heatmap_dir = Path(args.heatmap_dir).expanduser().resolve()
    out_dir     = Path(args.out_dir).expanduser().resolve()

    if not dtd_ocr_dir.is_dir():
        raise SystemExit(f"--dtd_ocr_dir not a directory: {dtd_ocr_dir}")
    if not heatmap_dir.is_dir():
        raise SystemExit(f"--heatmap_dir not a directory: {heatmap_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(dtd_ocr_dir.glob("*_dtd_ocr.json"))
    if args.limit > 0:
        json_files = json_files[: args.limit]
    print(f"[scan] {len(json_files)} dtd_ocr.json files")
    print(f"[heatmap_dir] {heatmap_dir}")
    print(f"[out] {out_dir}")

    counters = {"ok": 0, "skip_exists": 0, "no_heatmap": 0,
                "decode_err": 0, "exception": 0}
    for jp in tqdm(json_files, desc="annotating", unit="file"):
        try:
            status = annotate_one(jp, heatmap_dir, out_dir, args.overwrite)
        except Exception as exc:
            print(f"  [error] {jp.name}: {exc!r}", flush=True)
            status = "exception"
        counters[status] = counters.get(status, 0) + 1

    print(f"\n[done] {counters}")
    print(f"[done] outputs in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())