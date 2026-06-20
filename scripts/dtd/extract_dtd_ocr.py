#!/usr/bin/env python
"""Run DTD detection + OCR over a directory of images.

Produces {stem}_dtd_ocr.json with DTD regions, OCR output, and DTD-OCR overlaps.
Optionally renders annotated heatmaps with numbered DTD bounding boxes.
Use --ocr_only to run only OCR and skip DTD entirely.

Example:
    python scripts/dtd/extract_dtd_ocr.py \\
        --image_dir data/jpg \\
        --dtd_config config.yaml --dtd_checkpoint checkpoint.pth \\
        --out_json_dir data/dtd_ocr --out_heatmap_dir data/heatmaps \\
        --tta --skip_existing
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
# Path setup
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

from realtext_v2.grounding import mask_to_boxes  # noqa: E402

# DTD
_DTD_SCRIPT_DIR = _TOOLKIT_ROOT / "ForensicHub" / "dtd_train"
sys.path.insert(0, str(_DTD_SCRIPT_DIR))
import run_doc_forensics_inference as _dtd  # noqa: E402

# OCR
sys.path.insert(0, str(_TOOLKIT_ROOT / "scripts"))
from run_paddle_sobel import run_paddle_ocr_with_lang_detect  # noqa: E402


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _largest_internal_gap(present: np.ndarray, lo: int, hi: int):
    """Given a 1-D boolean occupancy array and the [lo, hi] span occupied by a
    component, return (gap_len, split_index) for the LONGEST run of empty
    cells strictly interior to (lo, hi). Returns (0, -1) if there is none.

    Used to find the blank band that separates two blobs that a connected-
    component pass lumped together."""
    best_len, best_split = 0, -1
    run_start = None
    for i in range(lo, hi + 1):
        if not present[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if run_start > lo and (i - 1) < hi:        # strictly interior
                    run_len = i - run_start
                    if run_len > best_len:
                        best_len = run_len
                        best_split = (run_start + i) // 2
                run_start = None
    return best_len, best_split


def _boxes_from_submask(sub: np.ndarray, ox: int, oy: int, min_area: int,
                        min_gap: int = 2, fill_thresh: float = 0.45,
                        depth: int = 0, max_depth: int = 8) -> list[list[int]]:
    """Recursively emit tight [x1, y1, x2, y2] boxes from one component's
    boolean sub-mask.

    A naive min/max box over a connected component swallows TWO distinct DTD
    regions whenever they are joined by a thin bridge or a diagonal touch
    (e.g. two stacked text lines becoming one tall box). To avoid that:

      * If the component fills most of its bounding box (a genuine solid blob)
        OR it is small / recursion is deep -> emit a single box.
      * Otherwise find the longest fully-empty interior band (row-wise giving a
        horizontal cut, or col-wise giving a vertical cut), cut the component
        there, and recurse on each half. This separates two blobs that only a
        thin connection had merged, examining the real extent on all four
        sides rather than just the outer corners."""
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
        # No clean gap to split on -> keep as a single box.
        if area >= min_area:
            return [[ox + x1, oy + y1, ox + x2 + 1, oy + y2 + 1]]
        return []

    out: list[list[int]] = []
    if rgap_len >= cgap_len:                    # horizontal cut at row rsplit
        top = sub.copy(); top[rsplit:, :] = False
        bot = sub.copy(); bot[:rsplit, :] = False
        out += _boxes_from_submask(top, ox, oy, min_area, min_gap,
                                   fill_thresh, depth + 1, max_depth)
        out += _boxes_from_submask(bot, ox, oy, min_area, min_gap,
                                   fill_thresh, depth + 1, max_depth)
    else:                                       # vertical cut at col csplit
        left  = sub.copy(); left[:, csplit:]  = False
        right = sub.copy(); right[:, :csplit] = False
        out += _boxes_from_submask(left,  ox, oy, min_area, min_gap,
                                   fill_thresh, depth + 1, max_depth)
        out += _boxes_from_submask(right, ox, oy, min_area, min_gap,
                                   fill_thresh, depth + 1, max_depth)
    return out


def _label_components(mask: np.ndarray, connectivity: int):
    """Return (n_labels, label_img) with label 0 == background. Prefers cv2,
    falls back to scipy.ndimage."""
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
    """Morphological opening to sever 1-2px bridges that 4-connectivity alone
    cannot break. ksize < 2 disables it. Real DTD blobs are far larger than
    the kernel, so they are preserved; only thin connectors are removed."""
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
    """Threshold the DTD probability map and return tight, NON-MERGED region
    boxes [x1, y1, x2, y2].

    Improvements over a plain min/max-of-connected-component box:
      1. Optional morphological opening severs thin bridges between blobs.
      2. 4-connectivity (default) so diagonally-touching blobs stay separate.
      3. Each component's full pixel extent is measured (all four sides).
      4. A low-fill component (two blobs joined by a thin link -> sparse
         bounding box) is split at its largest empty interior band, so two
         nearby DTD regions are emitted as TWO boxes instead of one swallowing
         rectangle.

    Boxes are returned sorted top-to-bottom, left-to-right for stable
    numbering."""
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

def _bbox_overlap_frac(box_a, box_b) -> float:
    """Fraction of box_b's area covered by box_a."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ax1, ax2 = sorted((ax1, ax2)); ay1, ay2 = sorted((ay1, ay2))
    bx1, bx2 = sorted((bx1, bx2)); by1, by2 = sorted((by1, by2))
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / b_area


def _compute_dtd_ocr_overlaps(
    dtd_regions: list,
    ocr_items: list,
    min_overlap_frac: float = 0.4,
) -> list[dict]:
    """For each DTD region, list overlapping OCR items.

    Returns list of dicts (one per region):
        {"region_index": 1, "bbox": [...], "overlapping_ocr": [
            {"id": .., "text": .., "bbox": .., "overlap_frac": 0.87}, ...
        ]}
    """
    out = []
    for i, box in enumerate(dtd_regions, start=1):
        try:
            x1, y1, x2, y2 = (int(v) for v in box)
        except (TypeError, ValueError):
            continue
        overlaps = []
        for it in ocr_items:
            ocr_bbox = it.get("bbox") or []
            if len(ocr_bbox) != 4:
                continue
            frac = _bbox_overlap_frac((x1, y1, x2, y2), ocr_bbox)
            if frac >= min_overlap_frac:
                overlaps.append({
                    "id": it.get("id"),
                    "text": it.get("text", ""),
                    "bbox": it.get("bbox"),
                    "overlap_frac": round(float(frac), 4),
                })
        overlaps.sort(key=lambda d: -d["overlap_frac"])
        out.append({
            "region_index": i,
            "bbox": [x1, y1, x2, y2],
            "overlapping_ocr": overlaps,
        })
    return out


# --------------------------------------------------------------------------- #
# Annotated heatmap renderer (matches annotate_heatmaps.py / infer_student.py)
# --------------------------------------------------------------------------- #
_BBOX_COLOR  = "#ff2e2e"
_HEATMAP_DPI = 150


def _render_annotated_heatmap(
    image_pil: Image.Image,
    prob: np.ndarray,
    dtd_regions: list,
    out_path: Path,
    prob_threshold: float = 0.40,
) -> None:
    """Build heatmap (jet overlay on image at SAME resolution as document) and
    draw numbered red bboxes. Only pixels with prob >= prob_threshold receive
    the warm-colour overlay; all others keep the original image pixel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    img_arr = np.asarray(image_pil.convert("RGB"))
    H, W = img_arr.shape[:2]
    if prob.shape != (H, W):
        prob_img = Image.fromarray(
            (prob * 255).clip(0, 255).astype(np.uint8)
        ).resize((W, H), Image.BILINEAR)
        prob = np.asarray(prob_img, dtype=np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    heat = (cmap(prob)[:, :, :3] * 255).astype(np.uint8)
    blended = (0.55 * img_arr + 0.45 * heat).clip(0, 255).astype(np.uint8)

    # Apply warm colours only where DTD probability exceeds the threshold.
    mask = prob >= prob_threshold
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    base = img_arr.copy()
    base[mask_3ch] = blended[mask_3ch]

    base_pil = Image.fromarray(base)
    hm_w, hm_h = base_pil.size

    fig_w_in = hm_w / _HEATMAP_DPI
    fig_h_in = hm_h / _HEATMAP_DPI
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=_HEATMAP_DPI)
    ax.set_xlim(0, hm_w)
    ax.set_ylim(hm_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(base_pil, extent=[0, hm_w, hm_h, 0])

    for i, box in enumerate(dtd_regions, start=1):
        try:
            x1, y1, x2, y2 = (int(v) for v in box)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))

        rect = mpatches.Rectangle(
            (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
            linewidth=2, edgecolor=_BBOX_COLOR, facecolor="none",
        )
        ax.add_patch(rect)
        bbox_h = max(1, y2 - y1)
        desired_text_px = 1.25 * bbox_h
        fontsize = max(6.0, desired_text_px * 72 / _HEATMAP_DPI * 0.75)
        ax.text(
            x1, max(0, y1 - 4), str(i),
            color="white", fontsize=fontsize, fontweight="bold",
            bbox=dict(facecolor=_BBOX_COLOR, edgecolor="none",
                       alpha=0.9, pad=1),
            zorder=5,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_HEATMAP_DPI, facecolor="white")
    plt.close(fig)


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
    dtd_model, dtd_model_name, dtd_needs_dct, device,
    ocr_engine: any,
    out_json_dir: Path,
    out_heatmap_dir: Path,
) -> str:
    """Returns status string ('ok' / 'skip_exists' / 'error')."""
    import torch

    stem = image_path.stem

    # Preserve subdirectory structure (e.g. part000/...)
    rel_dir = image_path.parent.relative_to(image_dir)
    if rel_dir == Path("."):
        rel_dir = Path("")

    json_dir = out_json_dir / rel_dir
    json_path = json_dir / f"{stem}_dtd_ocr.json"
    heatmap_path = None
    if not args.no_heatmap:
        heatmap_dir = out_heatmap_dir / rel_dir
        heatmap_path = heatmap_dir / f"{stem}.heatmap_annotated.png"

    if args.skip_existing and json_path.exists():
        if args.ocr_only or args.no_heatmap or (heatmap_path and heatmap_path.exists()):
            return "skip_exists"

    # ---- DTD inference ----
    dtd_regions = []
    prob = None
    image_pil = None
    if not args.ocr_only:
        tta_offsets = _dtd._build_tta_offsets(args.tta, args.tta_passes)
        try:
            prob, image_pil = _dtd.infer_one_image(
                image_path, dtd_model, dtd_model_name, dtd_needs_dct, device,
                jpeg_quality=args.jpeg_quality,
                tta_offsets=tta_offsets,
                tta_combine=args.tta_combine,
            )
        except Exception as exc:
            print(f"  [skip] DTD failed for {image_path.name}: {exc!r}", flush=True)
            return "error"

        dtd_regions = _dtd_regions_from_prob(
            prob, threshold=args.dtd_threshold, min_area=args.min_area,
            connectivity=args.cc_connectivity,
            morph_open_ksize=args.morph_open_ksize,
            split_low_fill=not args.no_split_low_fill,
            fill_split_thresh=args.fill_split_thresh,
            min_split_gap=args.min_split_gap,
        )
    else:
        image_pil = Image.open(image_path)

    orig_w, orig_h = image_pil.size

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

    # ---- Overlaps DTD <-> OCR ----
    region_overlaps = _compute_dtd_ocr_overlaps(
        dtd_regions, ocr_items,
        min_overlap_frac=args.min_overlap_frac,
    )

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
        "image_name":           image_path.name,
        "stem":                 stem,
        "image_size":           [orig_w, orig_h],
        "dtd_threshold":        args.dtd_threshold,
        "dtd_regions":          dtd_regions,
        "dtd_region_overlaps":  region_overlaps,
        "ocr_n_items":          n_items,
        "ocr_lang":             ocr_lang,
        "ocr_input":            ocr_input,
    }
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- Annotated heatmap ----
    if not args.no_heatmap:
        _render_annotated_heatmap(
            image_pil=image_pil, prob=prob, dtd_regions=dtd_regions,
            out_path=heatmap_path, prob_threshold=args.dtd_threshold,
        )

    if args.save_prob and prob is not None:
        prob_dir = Path(args.save_prob).expanduser().resolve() / rel_dir
        prob_dir.mkdir(parents=True, exist_ok=True)
        np.save(prob_dir / f"{stem}_dtd_prob.npy",
                prob.astype(np.float32))

    print(f"  [ok] {len(dtd_regions)} DTD regions, "
          f"{n_items} OCR items ({ocr_lang})")
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
                    help="Output dir for {stem}_dtd_ocr.json files.")
    ap.add_argument("--out_heatmap_dir", default=None,
                    help="Output dir for {stem}.heatmap_annotated.png files. "
                         "Required unless --no_heatmap or --ocr_only is set.")

    # Mode
    ap.add_argument("--ocr_only", action="store_true",
                    help="Run ONLY OCR on images and save only OCR output in "
                         "JSON files. DTD inference, heatmaps, and probability "
                         "maps are skipped. --dtd_config, --dtd_checkpoint, "
                         "and --out_heatmap_dir are NOT required in this mode.")

    # DTD
    ap.add_argument("--dtd_config", default=None, help="DTD YAML config.")
    ap.add_argument("--dtd_checkpoint", default=None, help="DTD .pth.")
    ap.add_argument("--dtd_threshold", type=float, default=0.40,
                    help="DTD probability threshold for region bboxes.")
    ap.add_argument("--min_area", type=int, default=200,
                    help="Min connected-component area (pixels) for a region.")
    ap.add_argument("--cc_connectivity", type=int, default=4, choices=(4, 8),
                    help="Connected-component connectivity for region "
                         "extraction. 4 (default) keeps diagonally-touching "
                         "blobs separate; 8 merges them.")
    ap.add_argument("--morph_open_ksize", type=int, default=3,
                    help="Kernel size for morphological opening applied to the "
                         "thresholded mask before labelling, to sever thin "
                         "bridges between distinct regions. 0/1 disables it.")
    ap.add_argument("--no_split_low_fill", action="store_true",
                    help="Disable splitting of sparse (low-fill) components. "
                         "By default a component whose pixels fill less than "
                         "--fill_split_thresh of its bounding box is split at "
                         "its largest empty interior band, so two nearby DTD "
                         "regions are not swallowed by one box.")
    ap.add_argument("--fill_split_thresh", type=float, default=0.45,
                    help="A component is considered for splitting when its "
                         "occupied-pixel fraction of its bounding box is below "
                         "this value.")
    ap.add_argument("--min_split_gap", type=int, default=2,
                    help="Minimum width (pixels) of an empty interior band "
                         "required to split a component there.")
    ap.add_argument("--jpeg_quality", type=int, default=95,
                    help="JPEG quality used during DTD inference.")
    ap.add_argument("--save_prob", default=None, metavar="PROB_DIR",
                    help="If set to a directory path, also save the raw DTD "
                         "probability map for each image as "
                         "{PROB_DIR}/{partXXX}/{stem}_dtd_prob.npy (float32). "
                         "The part-subfolder structure is mirrored. Omit to "
                         "skip saving probability maps.")

    # TTA
    ap.add_argument("--tta", action="store_true",
                    help="Enable test-time augmentation (multiple DTD passes "
                         "with shifted crop grids).")
    ap.add_argument("--tta_combine", default="min",
                    choices=("min", "mean", "median"),
                    help="How to combine per-pass probability maps: "
                         "min=conservative, mean=smoothing, median=middle.")
    ap.add_argument("--tta_passes", type=int, default=4, choices=(3, 4),
                    help="Number of TTA passes (3 or 4).")

    # OCR
    ap.add_argument("--langs", default="en,ch,th,ms,id,ar",
                    help="Comma-separated PaddleOCR language codes for "
                         "language detection.")

    # Overlap
    ap.add_argument("--min_overlap_frac", type=float, default=0.4,
                    help="Minimum fraction of an OCR bbox's area that must "
                         "be covered by a DTD region for them to count as "
                         "overlapping. Matches prerender_prompts.py default.")

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
                    help="Skip images whose JSON + heatmap both already exist.")
    ap.add_argument("--no_heatmap", action="store_true",
                    help="Do not save annotated heatmap PNGs. "
                         "Only JSON (and optionally --save_prob) are written.")
    ap.add_argument("--skip_png", action="store_true",
                    help="Skip all .png files (do not run DTD or OCR on them).")

    # Internal — used by spawned workers, not by users.
    ap.add_argument("--_worker_gpu_id", type=int, default=-1,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.ocr_only:
        if not args.no_heatmap:
            args.no_heatmap = True
    else:
        if args.dtd_config is None:
            ap.error("--dtd_config is required (unless --ocr_only)")
        if args.dtd_checkpoint is None:
            ap.error("--dtd_checkpoint is required (unless --ocr_only)")
        if not args.no_heatmap and args.out_heatmap_dir is None:
            ap.error("--out_heatmap_dir is required unless --no_heatmap or --ocr_only is set")
    return args


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
    CUDA import / DTD load). Processes the given shard of image paths.

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

    image_dir       = Path(args.image_dir).expanduser().resolve()
    out_json_dir    = Path(args.out_json_dir).expanduser().resolve()
    out_heatmap_dir = (Path(args.out_heatmap_dir).expanduser().resolve()
                       if args.out_heatmap_dir else Path("."))

    if args.ocr_only:
        dtd_model = dtd_model_name = dtd_needs_dct = None
        print(f"[worker gpu={gpu_id}] OCR-only mode: skipping DTD load",
              flush=True)
        if not torch.cuda.is_available():
            print(f"[worker gpu={gpu_id}] WARNING: CUDA not available; "
                  f"PaddleOCR will run on CPU", flush=True)
    else:
        # Load DTD on this worker
        t0 = time.time()
        _dtd._setup_paths_and_registry()
        dtd_model, dtd_model_name, dtd_needs_dct = _dtd.build_model_and_load(
            args.dtd_config, args.dtd_checkpoint, device,
        )
        print(f"[worker gpu={gpu_id}] dtd loaded ({dtd_model_name}) "
              f"in {time.time()-t0:.1f}s", flush=True)

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
                    dtd_model=dtd_model, dtd_model_name=dtd_model_name,
                    dtd_needs_dct=dtd_needs_dct, device=device,
                    ocr_engine=ocr_engine,
                    out_json_dir=out_json_dir,
                    out_heatmap_dir=out_heatmap_dir,
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
    if not args.ocr_only:
        out_heatmap_dir = Path(args.out_heatmap_dir).expanduser().resolve()
        out_heatmap_dir.mkdir(parents=True, exist_ok=True)
        if args.save_prob:
            Path(args.save_prob).expanduser().resolve().mkdir(
                parents=True, exist_ok=True)
    else:
        out_heatmap_dir = Path(".")

    image_paths = gather_images(image_dir, args.limit, skip_png=args.skip_png)
    if not image_paths:
        print("[error] no images found")
        return 1
    if args.skip_png:
        print("[run] skipping .png files (--skip_png)")
    if args.ocr_only:
        print(f"[run] OCR-ONLY mode  {len(image_paths)} image(s) from {image_dir}")
    else:
        print(f"[run] {len(image_paths)} image(s) from {image_dir}")
    print(f"[out] json     -> {out_json_dir}")
    if not args.ocr_only:
        print(f"[out] heatmap  -> {out_heatmap_dir}")

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
            name=f"extractor-gpu{gpu_id}",
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
                    gpu_id = int(p.name.replace("extractor-gpu", ""))
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