#!/usr/bin/env python
"""Augment GT forensic reports with DTD/OCR/mask evidence.

For each {stem}_dtd_ocr.json, produces five augmented report variants by
refining GT grounding coordinates against DTD regions, OCR word-bboxes, and
GT segmentation masks. Generates TP/FP labels for DTD regions and writes
all augmentation logs back into the JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# =========================================================================== #
# Split-aware region extraction (copied verbatim from extract_dtd_ocr.py so
# this script is standalone and applies the IDENTICAL algorithm to the GT mask)
# =========================================================================== #
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


# =========================================================================== #
# Geometry helpers (IoU + intersection-over-min, matching distill_cot_235b.py)
# =========================================================================== #
def _bbox_iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_intersection_over_min(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def _union_box(boxes):
    xs1 = min(b[0] for b in boxes); ys1 = min(b[1] for b in boxes)
    xs2 = max(b[2] for b in boxes); ys2 = max(b[3] for b in boxes)
    return (xs1, ys1, xs2, ys2)


# =========================================================================== #
# GT mask -> split-aware bboxes
# =========================================================================== #
def mask_to_bboxes_split(mask_path: Path, *, min_area: int = 200,
                         connectivity: int = 4, morph_open_ksize: int = 3,
                         fill_split_thresh: float = 0.45,
                         min_split_gap: int = 2) -> list[tuple]:
    """Extract GT bboxes from a binary tampering mask using the SAME
    split-aware algorithm as extract_dtd_ocr_heatmaps.py (touching regions
    are split at the largest empty interior band)."""
    m = Image.open(str(mask_path)).convert("L")
    arr = np.array(m, dtype=np.uint8)
    prob = (arr > 0).astype(np.float32)   # binary mask as a 0/1 "prob" map
    if not prob.any():
        return []
    boxes = _dtd_regions_from_prob(
        prob, threshold=0.5, min_area=min_area,
        connectivity=connectivity, morph_open_ksize=morph_open_ksize,
        split_low_fill=True, fill_split_thresh=fill_split_thresh,
        min_split_gap=min_split_gap,
    )
    return [tuple(int(v) for v in b) for b in boxes]


# =========================================================================== #
# TP / FP for DTD regions vs GT mask bboxes
# =========================================================================== #
def classify_dtd_regions(dtd_boxes, mask_bboxes, *,
                         iou_threshold: float = 0.05) -> list[bool]:
    """A DTD region is TP if it overlaps ANY GT-mask bbox (intersection-over-
    min >= iou_threshold), else FP. Matches distill_cot_235b.py."""
    out = []
    for d in dtd_boxes:
        is_tp = any(_bbox_intersection_over_min(d, mb) >= iou_threshold
                    for mb in mask_bboxes)
        out.append(bool(is_tp))
    return out


# =========================================================================== #
# Report augmentation
# =========================================================================== #
_GROUNDING_RE = re.compile(
    r"\[GROUNDING\]:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)
# An anomaly block: from "### ANOMALY" up to (but not including) the next
# "### " heading or the SUMMARY / end-of-report marker.
_ANOMALY_BLOCK_RE = re.compile(
    r"(###\s+ANOMALY.*?)(?=###\s+ANOMALY|\n##\s|\*\*END OF REPORT\*\*|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _replace_first_grounding(block: str, new_box) -> str:
    """Replace the first [GROUNDING] coords in an anomaly block."""
    repl = (f"[GROUNDING]: [{int(new_box[0])}, {int(new_box[1])}, "
            f"{int(new_box[2])}, {int(new_box[3])}]")
    return _GROUNDING_RE.sub(repl, block, count=1)


def _first_grounding(block: str):
    m = _GROUNDING_RE.search(block)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)),
            int(m.group(3)), int(m.group(4)))


# ----- Report 1: refine GT groundings against GT-mask bboxes ----- #
def refine_report_with_mask(gt_report_text: str, mask_bboxes, *,
                            min_overlap: float = 0.10):
    """Replace each [GROUNDING] bbox with the best-overlapping GT-mask bbox
    (intersection-over-min >= min_overlap). Returns (text, log)."""
    log = []

    def _replace(m: re.Match) -> str:
        coords = (int(m.group(1)), int(m.group(2)),
                  int(m.group(3)), int(m.group(4)))
        if not mask_bboxes:
            log.append({"original": list(coords), "replaced_with": None,
                        "reason": "no mask bboxes"})
            return m.group(0)
        scores = sorted(((_bbox_intersection_over_min(coords, mb), mb)
                         for mb in mask_bboxes), reverse=True, key=lambda t: t[0])
        best_score, best_mb = scores[0]
        if best_score >= min_overlap:
            log.append({"original": list(coords),
                        "replaced_with": list(best_mb),
                        "score": round(float(best_score), 4)})
            return (f"[GROUNDING]: [{best_mb[0]}, {best_mb[1]}, "
                    f"{best_mb[2]}, {best_mb[3]}]")
        log.append({"original": list(coords), "replaced_with": None,
                    "reason": f"best overlap={best_score:.3f}"})
        return m.group(0)

    refined = _GROUNDING_RE.sub(_replace, gt_report_text)
    return refined, log


# ----- helper: best OCR-union box for a grounding (IoU based) ----- #
def _best_ocr_box(coords, ocr_items, *, iou_thresh: float):
    """Return (box, iou, n_overlapping, overlapping_items) for the best OCR
    match to `coords`.  Tries every single OCR box and every pair union,
    picks the one with highest IoU.  Only candidates with IoU > iou_thresh
    are considered."""
    candidates = []
    for it in ocr_items:
        bb = it.get("bbox")
        if not bb or len(bb) != 4:
            continue
        bb_t = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        iou = _bbox_iou(coords, bb_t)
        if iou > iou_thresh:
            candidates.append({"item": it, "bbox": bb_t, "iou": iou})
    if not candidates:
        return None, 0.0, 0, []

    best = None
    best_iou = 0.0
    for c in candidates:
        if c["iou"] > best_iou:
            best = c
            best_iou = c["iou"]

    if len(candidates) >= 2:
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                union = _union_box([candidates[i]["bbox"], candidates[j]["bbox"]])
                union_iou = _bbox_iou(coords, union)
                if union_iou > best_iou:
                    best = {
                        "bbox": union, "iou": union_iou,
                        "pair": [candidates[i], candidates[j]]}
                    best_iou = union_iou

    if "pair" in best:
        pair = best["pair"]
        overlapping = []
        for p in pair:
            it = p["item"]
            overlapping.append({
                "id": it.get("id"),
                "text": str(it.get("text", "")).replace('"', "'"),
                "bbox": list(p["bbox"]),
                "iou": round(p["iou"], 4),
            })
        return list(best["bbox"]), best_iou, 2, overlapping
    else:
        it = best["item"]
        return list(best["bbox"]), best_iou, 1, [
            {"id": it.get("id"),
             "text": str(it.get("text", "")).replace('"', "'"),
             "bbox": list(best["bbox"]),
             "iou": round(best["iou"], 4)}
        ]


def _best_dtd_box(coords, dtd_boxes, *, iou_thresh: float):
    """Return (box, iou, n_overlapping, overlapping_items) for the best DTD
    match to `coords`.  Tries every single DTD region and every pair union,
    picks the one with highest IoU.  Only candidates with IoU > iou_thresh
    are considered."""
    candidates = []
    for d in dtd_boxes:
        d_t = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        iou = _bbox_iou(coords, d_t)
        if iou > iou_thresh:
            candidates.append({"bbox": d_t, "iou": iou})
    if not candidates:
        return None, 0.0, 0, []

    best = None
    best_iou = 0.0
    for c in candidates:
        if c["iou"] > best_iou:
            best = c
            best_iou = c["iou"]

    if len(candidates) >= 2:
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                union = _union_box([candidates[i]["bbox"], candidates[j]["bbox"]])
                union_iou = _bbox_iou(coords, union)
                if union_iou > best_iou:
                    best = {
                        "bbox": union, "iou": union_iou,
                        "pair": [candidates[i], candidates[j]]}
                    best_iou = union_iou

    if "pair" in best:
        pair = best["pair"]
        return list(best["bbox"]), best_iou, 2, [
            {"bbox": list(pair[0]["bbox"]), "iou": round(pair[0]["iou"], 4)},
            {"bbox": list(pair[1]["bbox"]), "iou": round(pair[1]["iou"], 4)},
        ]
    else:
        return list(best["bbox"]), best_iou, 1, [
            {"bbox": list(best["bbox"]), "iou": round(best["iou"], 4)}
        ]


# ----- Reports 2/3/4 operate per anomaly block on the REFINED report ----- #
def build_report_2_ocr(refined_text, ocr_items, *, iou_thresh: float):
    """Report 2: replace grounding with matching OCR-union box (IoU>thresh);
    no overlap -> keep refined coords."""
    log = []

    def _per_block(m: re.Match) -> str:
        block = m.group(1)
        coords = _first_grounding(block)
        if coords is None:
            return block
        ocr_box, iou, n_overlapping, overlapping = _best_ocr_box(
            coords, ocr_items, iou_thresh=iou_thresh)
        if ocr_box is not None:
            if n_overlapping == 1:
                item = overlapping[0]
                entry = {"grounding": list(coords), "source": "ocr",
                         "n_overlapping_ocr": 1,
                         "id": item.get("id"),
                         "text": item.get("text", ""),
                         "bbox": item.get("bbox"),
                         "iou": item.get("iou")}
            else:
                entry = {"grounding": list(coords), "source": "ocr",
                         "new": list(ocr_box), "iou": round(iou, 4),
                         "n_overlapping_ocr": n_overlapping,
                         "overlapping_ocr_items": overlapping}
            if n_overlapping > 1:
                print(f"    [MULTI_OCR] {n_overlapping} OCR boxes overlap "
                      f"grounding {coords} -> union {list(ocr_box)}", flush=True)
            log.append(entry)
            return _replace_first_grounding(block, ocr_box)
        log.append({"grounding": list(coords), "source": "kept",
                    "new": list(coords)})
        return block

    out = _ANOMALY_BLOCK_RE.sub(_per_block, refined_text)
    return out, log


def build_report_3_ocr_dtd(refined_text, ocr_items, dtd_boxes, *,
                           iou_thresh: float):
    """Report 3: like report 2 but DTD regions take PRIORITY. If a grounding
    overlaps both OCR and a DTD region, use the DTD region's coords; else OCR;
    else keep refined coords."""
    log = []

    def _per_block(m: re.Match) -> str:
        block = m.group(1)
        coords = _first_grounding(block)
        if coords is None:
            return block
        dtd_box, dtd_iou, n_dtd, dtd_items = _best_dtd_box(
            coords, dtd_boxes, iou_thresh=iou_thresh)
        ocr_box, ocr_iou, n_ocr, ocr_items_list = _best_ocr_box(
            coords, ocr_items, iou_thresh=iou_thresh)
        if dtd_box is not None:
            entry = {"grounding": list(coords), "source": "dtd",
                     "new": list(dtd_box), "iou": round(dtd_iou, 4),
                     "n_overlapping_dtd": n_dtd}
            if n_dtd > 1:
                entry["overlapping_dtd_items"] = dtd_items
                print(f"    [MULTI_DTD] {n_dtd} DTD boxes overlap "
                      f"grounding {coords} -> union {list(dtd_box)}", flush=True)
            log.append(entry)
            return _replace_first_grounding(block, dtd_box)
        if ocr_box is not None:
            if n_ocr == 1:
                item = ocr_items_list[0]
                entry = {"grounding": list(coords), "source": "ocr",
                         "n_overlapping_ocr": 1,
                         "id": item.get("id"),
                         "text": item.get("text", ""),
                         "bbox": item.get("bbox"),
                         "iou": item.get("iou")}
            else:
                entry = {"grounding": list(coords), "source": "ocr",
                         "new": list(ocr_box), "iou": round(ocr_iou, 4),
                         "n_overlapping_ocr": n_ocr,
                         "overlapping_ocr_items": ocr_items_list}
            if n_ocr > 1:
                print(f"    [MULTI_OCR] {n_ocr} OCR boxes overlap "
                      f"grounding {coords} -> union {list(ocr_box)}", flush=True)
            log.append(entry)
            return _replace_first_grounding(block, ocr_box)
        log.append({"grounding": list(coords), "source": "kept",
                    "new": list(coords)})
        return block

    out = _ANOMALY_BLOCK_RE.sub(_per_block, refined_text)
    return out, log


def build_report_4_dtd_only(refined_text, dtd_boxes, *, iou_thresh: float):
    """Report 4: keep ONLY anomalies whose grounding overlaps a DTD region
    (IoU>thresh), replacing coords with the DTD region's. Drop the rest.
    Anomalies are renumbered ANOMALY_001, 002, ... after dropping."""
    log = []
    kept_blocks = []

    for m in _ANOMALY_BLOCK_RE.finditer(refined_text):
        block = m.group(1)
        coords = _first_grounding(block)
        if coords is None:
            log.append({"grounding": None, "kept": False, "reason": "no grounding"})
            continue
        dtd_box, dtd_iou, n_dtd, dtd_items = _best_dtd_box(
            coords, dtd_boxes, iou_thresh=iou_thresh)
        if dtd_box is None:
            log.append({"grounding": list(coords), "kept": False,
                        "reason": "no DTD overlap"})
            continue
        new_block = _replace_first_grounding(block, dtd_box)
        kept_blocks.append(new_block)
        entry = {"grounding": list(coords), "kept": True,
                 "new": list(dtd_box), "iou": round(dtd_iou, 4),
                 "n_overlapping_dtd": n_dtd}
        if n_dtd > 1:
            entry["overlapping_dtd_items"] = dtd_items
        log.append(entry)

    # Reassemble: header (before first anomaly) + kept anomaly blocks +
    # tail (SUMMARY ... END OF REPORT). Renumber ANOMALY_NNN.
    first = _ANOMALY_BLOCK_RE.search(refined_text)
    header = refined_text[:first.start()] if first else refined_text
    # tail = everything from SUMMARY / END marker onward
    tail = ""
    mtail = re.search(r"(\n##\s+SUMMARY.*)$", refined_text,
                      re.IGNORECASE | re.DOTALL)
    if mtail:
        tail = mtail.group(1)
    else:
        mend = re.search(r"(\*\*END OF REPORT\*\*\s*)$", refined_text,
                         re.IGNORECASE | re.DOTALL)
        tail = ("\n" + mend.group(1)) if mend else ""

    renum = []
    for i, blk in enumerate(kept_blocks, start=1):
        blk2 = re.sub(r"ANOMALY[_ ]?\d+",
                      f"ANOMALY_{i:03d}", blk, count=1, flags=re.IGNORECASE)
        renum.append(blk2.rstrip())
    body = "\n\n".join(renum)
    out = header.rstrip() + ("\n\n" if renum else "\n") + body
    if tail:
        out = out.rstrip() + "\n" + tail.lstrip("\n")
    return out, log, len(kept_blocks)


# ----- Report ocr_only: keep only anomalies with OCR overlap, drop rest ----- #
def build_report_ocr_only(refined_text, ocr_items, *, iou_thresh: float):
    """Report ocr_only: keep ONLY anomalies whose grounding overlaps an OCR word
    box (IoU>iou_thresh), replacing coords with the OCR box. Drop the rest.
    Anomalies are renumbered ANOMALY_001, 002, ... after dropping."""
    log = []
    kept_blocks = []

    for m in _ANOMALY_BLOCK_RE.finditer(refined_text):
        block = m.group(1)
        coords = _first_grounding(block)
        if coords is None:
            log.append({"grounding": None, "kept": False, "reason": "no grounding"})
            continue
        ocr_box, ocr_iou, n_overlapping, overlapping = _best_ocr_box(
            coords, ocr_items, iou_thresh=iou_thresh)
        if ocr_box is None:
            log.append({"grounding": list(coords), "kept": False,
                        "reason": "no OCR overlap"})
            continue
        new_block = _replace_first_grounding(block, ocr_box)
        kept_blocks.append(new_block)
        if n_overlapping == 1:
            item = overlapping[0]
            entry = {"grounding": list(coords), "kept": True,
                     "n_overlapping_ocr": 1,
                     "id": item.get("id"),
                     "text": item.get("text", ""),
                     "bbox": item.get("bbox"),
                     "iou": item.get("iou")}
        else:
            entry = {"grounding": list(coords), "kept": True,
                     "new": list(ocr_box), "iou": round(ocr_iou, 4),
                     "n_overlapping_ocr": n_overlapping,
                     "overlapping_ocr_items": overlapping}
        if n_overlapping > 1:
            print(f"    [MULTI_OCR] {n_overlapping} OCR boxes overlap "
                  f"grounding {coords} -> union {list(ocr_box)}", flush=True)
        log.append(entry)

    # Reassemble: header (before first anomaly) + kept anomaly blocks +
    # tail (SUMMARY / END OF REPORT). Renumber ANOMALY_NNN.
    first = _ANOMALY_BLOCK_RE.search(refined_text)
    header = refined_text[:first.start()] if first else refined_text
    tail = ""
    mtail = re.search(r"(\n##\s+SUMMARY.*)$", refined_text,
                      re.IGNORECASE | re.DOTALL)
    if mtail:
        tail = mtail.group(1)
    else:
        mend = re.search(r"(\*\*END OF REPORT\*\*\s*)$", refined_text,
                         re.IGNORECASE | re.DOTALL)
        tail = ("\n" + mend.group(1)) if mend else ""

    renum = []
    for i, blk in enumerate(kept_blocks, start=1):
        blk2 = re.sub(r"ANOMALY[_ ]?\d+",
                      f"ANOMALY_{i:03d}", blk, count=1, flags=re.IGNORECASE)
        renum.append(blk2.rstrip())
    body = "\n\n".join(renum)
    out = header.rstrip() + ("\n\n" if renum else "\n") + body
    if tail:
        out = out.rstrip() + "\n" + tail.lstrip("\n")
    return out, log, len(kept_blocks)


# ----- Report ocr_dtd_only: keep anomalies with OCR OR DTD overlap ----- #
def build_report_ocr_dtd_only(refined_text, ocr_items, dtd_boxes, *,
                              iou_thresh: float):
    """Report ocr_dtd_only: keep ONLY anomalies whose grounding overlaps
    EITHER a DTD region OR an OCR word box (IoU>thresh), replacing coords with
    the best match (DTD takes priority). Drop the rest.
    Anomalies are renumbered ANOMALY_001, 002, ... after dropping."""
    log = []
    kept_blocks = []

    for m in _ANOMALY_BLOCK_RE.finditer(refined_text):
        block = m.group(1)
        coords = _first_grounding(block)
        if coords is None:
            log.append({"grounding": None, "kept": False,
                        "reason": "no grounding"})
            continue
        dtd_box, dtd_iou, n_dtd, dtd_items = _best_dtd_box(
            coords, dtd_boxes, iou_thresh=iou_thresh)
        if dtd_box is not None:
            new_block = _replace_first_grounding(block, dtd_box)
            kept_blocks.append(new_block)
            entry = {"grounding": list(coords), "kept": True,
                     "source": "dtd",
                     "new": list(dtd_box), "iou": round(dtd_iou, 4),
                     "n_overlapping_dtd": n_dtd}
            if n_dtd > 1:
                entry["overlapping_dtd_items"] = dtd_items
            log.append(entry)
            continue
        ocr_box, ocr_iou, n_ocr, ocr_items_list = _best_ocr_box(
            coords, ocr_items, iou_thresh=iou_thresh)
        if ocr_box is not None:
            new_block = _replace_first_grounding(block, ocr_box)
            kept_blocks.append(new_block)
            if n_ocr == 1:
                item = ocr_items_list[0]
                entry = {"grounding": list(coords), "kept": True,
                         "source": "ocr",
                         "n_overlapping_ocr": 1,
                         "id": item.get("id"),
                         "text": item.get("text", ""),
                         "bbox": item.get("bbox"),
                         "iou": item.get("iou")}
            else:
                entry = {"grounding": list(coords), "kept": True,
                         "source": "ocr",
                         "new": list(ocr_box), "iou": round(ocr_iou, 4),
                         "n_overlapping_ocr": n_ocr,
                         "overlapping_ocr_items": ocr_items_list}
            log.append(entry)
            continue
        log.append({"grounding": list(coords), "kept": False,
                    "reason": "no OCR or DTD overlap"})

    # Reassemble: header + kept anomaly blocks + tail. Renumber ANOMALY_NNN.
    first = _ANOMALY_BLOCK_RE.search(refined_text)
    header = refined_text[:first.start()] if first else refined_text
    tail = ""
    mtail = re.search(r"(\n##\s+SUMMARY.*)$", refined_text,
                      re.IGNORECASE | re.DOTALL)
    if mtail:
        tail = mtail.group(1)
    else:
        mend = re.search(r"(\*\*END OF REPORT\*\*\s*)$", refined_text,
                         re.IGNORECASE | re.DOTALL)
        tail = ("\n" + mend.group(1)) if mend else ""

    renum = []
    for i, blk in enumerate(kept_blocks, start=1):
        blk2 = re.sub(r"ANOMALY[_ ]?\d+",
                      f"ANOMALY_{i:03d}", blk, count=1, flags=re.IGNORECASE)
        renum.append(blk2.rstrip())
    body = "\n\n".join(renum)
    out = header.rstrip() + ("\n\n" if renum else "\n") + body
    if tail:
        out = out.rstrip() + "\n" + tail.lstrip("\n")
    return out, log, len(kept_blocks)


# =========================================================================== #
# Path helpers
# =========================================================================== #
def find_gt_report(reports_dir: Path, stem: str) -> Optional[Path]:
    # 1. Flat search (legacy layout, no partXXX subdirs)
    for suffix in ("_report.md", ".md", "_report.txt", ".txt"):
        p = reports_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    # 2. Recursive search inside partXXX subfolders
    for suffix in ("_report.md", ".md", "_report.txt", ".txt"):
        matches = list(reports_dir.rglob(f"{stem}{suffix}"))
        if matches:
            return matches[0]
    return None


def find_gt_mask(masks_dir: Path, stem: str) -> Optional[Path]:
    # 1. Flat search (legacy layout, no partXXX subdirs)
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
        for cand in (masks_dir / f"{stem}_mask{ext}",
                     masks_dir / f"{stem}{ext}"):
            if cand.exists():
                return cand
    # 2. Recursive search inside partXXX subfolders
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
        for pattern in (f"{stem}_mask{ext}", f"{stem}{ext}"):
            matches = list(masks_dir.rglob(pattern))
            if matches:
                return matches[0]
    return None


# =========================================================================== #
# Visualization (optional, --viz_dir)
# =========================================================================== #
def _all_groundings(report_text: str) -> list[tuple]:
    """All [GROUNDING] boxes in a report, in order."""
    out = []
    for m in _GROUNDING_RE.finditer(report_text):
        out.append((int(m.group(1)), int(m.group(2)),
                    int(m.group(3)), int(m.group(4))))
    return out


def _find_image(image_dir: Optional[Path], image_name: str,
                stem: str) -> Optional[Path]:
    if image_dir is None:
        return None
    cands = [image_dir / image_name]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        cands.append(image_dir / f"{stem}{ext}")
    # also try part-subdir recursive (image_dir may hold partXXX folders)
    for ext in (".jpg", ".jpeg", ".png"):
        cands += list(image_dir.rglob(f"{stem}{ext}"))
    for c in cands:
        if c.exists():
            return c
    return None


# Box layers: (label, color, list-getter, linewidth, draw-numbers)
_VIZ_COLORS = {
    "gt_mask":   "#19c819",   # green
    "dtd_tp":    "#1f6bff",   # blue
    "dtd_fp":    "#ff2e2e",   # red
    "ocr":       "#ff9f1c",   # orange (thin)
    "report_1":  "#b14aed",   # purple
    "report_2":  "#00c2d1",   # cyan
    "report_3":  "#f7d000",   # yellow
    "report_4":  "#15d6a0",   # teal
    "ocr_only":  "#e6194b",   # bright red
    "ocr_dtd_only": "#ff00ff", # magenta
}


def _visualize(out_png: Path, image_pil, image_size,
               *, mask_bboxes, dtd_boxes, tp_flags, ocr_items,
               reports: dict, draw_ocr: bool = True) -> None:
    """Draw every box layer in its own colour onto a multi-panel figure.

    Panel A: evidence (GT mask / DTD TP / DTD FP / OCR words).
    Panel B: the four report groundings, each report its own colour.
    Drawing on the real image if available, else on a white canvas sized to
    image_size [W, H]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if image_pil is not None:
        base = np.asarray(image_pil.convert("RGB"))
        H, W = base.shape[:2]
    else:
        W, H = (image_size or [1000, 1000])
        base = np.full((int(H), int(W), 3), 255, dtype=np.uint8)

    def _draw(ax, boxes, color, lw=2, numbers=False, alpha=1.0):
        for i, b in enumerate(boxes, start=1):
            if not b or len(b) != 4:
                continue
            x1, y1, x2, y2 = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
                linewidth=lw, edgecolor=color, facecolor="none", alpha=alpha))
            if numbers:
                ax.text(x1, max(0, y1 - 3), str(i), color="white",
                        fontsize=8, fontweight="bold",
                        bbox=dict(facecolor=color, edgecolor="none",
                                  alpha=0.85, pad=1), zorder=6)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, W / 90 * 2), max(6, H / 90)))

    # --- Panel A: evidence ---
    axA = axes[0]
    axA.imshow(base); axA.axis("off")
    axA.set_title("Evidence: GT-mask(green)  DTD-TP(blue)  DTD-FP(red)  "
                  "OCR(orange)", fontsize=9)
    _draw(axA, mask_bboxes, _VIZ_COLORS["gt_mask"], lw=2.5, numbers=True)
    tp_boxes = [d for d, f in zip(dtd_boxes, tp_flags) if f]
    fp_boxes = [d for d, f in zip(dtd_boxes, tp_flags) if not f]
    _draw(axA, tp_boxes, _VIZ_COLORS["dtd_tp"], lw=2)
    _draw(axA, fp_boxes, _VIZ_COLORS["dtd_fp"], lw=2)
    if draw_ocr:
        ocr_boxes = [it.get("bbox") for it in ocr_items
                     if it.get("bbox") and len(it["bbox"]) == 4]
        _draw(axA, ocr_boxes, _VIZ_COLORS["ocr"], lw=0.6, alpha=0.6)

    # --- Panel B: report groundings ---
    axB = axes[1]
    axB.imshow(base); axB.axis("off")
    axB.set_title("Groundings: R1-refine(purple)  R2-ocr(cyan)  "
                  "R3-ocr+dtd(yellow)  R4-dtd-only(teal)  "
                  "OCR-only(red)  OCR-DTD-only(magenta)", fontsize=9)
    _draw(axB, _all_groundings(reports.get("report_1_refined", "")),
          _VIZ_COLORS["report_1"], lw=3.0, numbers=True)
    _draw(axB, _all_groundings(reports.get("report_2_ocr", "")),
          _VIZ_COLORS["report_2"], lw=2.2)
    _draw(axB, _all_groundings(reports.get("report_3_ocr_dtd", "")),
          _VIZ_COLORS["report_3"], lw=1.6)
    _draw(axB, _all_groundings(reports.get("report_4_dtd_only", "")),
          _VIZ_COLORS["report_4"], lw=1.0)
    _draw(axB, _all_groundings(reports.get("ocr_only", "")),
          _VIZ_COLORS["ocr_only"], lw=0.8)
    _draw(axB, _all_groundings(reports.get("ocr_dtd_only", "")),
          _VIZ_COLORS["ocr_dtd_only"], lw=0.6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================================================================== #
# Per-JSON processing
# =========================================================================== #
def process_one_json(json_path: Path, *, args,
                     gt_masks_dir: Path, gt_reports_dir: Path) -> str:
    """Augment one {stem}_dtd_ocr.json in place. Returns status string."""
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [error] cannot read {json_path.name}: {exc!r}", flush=True)
        return "error"

    stem = payload.get("stem") or json_path.name.replace("_dtd_ocr.json", "")

    if args.skip_existing and payload.get("augmentation") is not None:
        return "skip_exists"

    # ---- DTD regions + OCR items from the JSON ----
    dtd_boxes = [tuple(int(v) for v in b) for b in payload.get("dtd_regions", [])]
    ocr_items = (payload.get("ocr_input", {}) or {}).get("ocr_items", []) or []

    # ---- GT mask -> split-aware bboxes ----
    gt_mask_path = find_gt_mask(gt_masks_dir, stem)
    if gt_mask_path is None:
        print(f"  [skip] no GT mask for {stem}", flush=True)
        return "no_mask"
    mask_bboxes = mask_to_bboxes_split(
        gt_mask_path, min_area=args.mask_min_area,
        connectivity=args.cc_connectivity,
        morph_open_ksize=args.morph_open_ksize,
        min_split_gap=args.min_split_gap,
    )

    # ---- TP / FP for DTD regions ----
    tp_flags = classify_dtd_regions(dtd_boxes, mask_bboxes,
                                    iou_threshold=args.tp_iou)
    n_tp = sum(tp_flags); n_fp = len(tp_flags) - n_tp

    # ---- GT report ----
    gt_report_path = find_gt_report(gt_reports_dir, stem)
    if gt_report_path is None:
        print(f"  [skip] no GT report for {stem}", flush=True)
        return "no_report"
    gt_report_text = gt_report_path.read_text(encoding="utf-8").strip()

    # ---- Report 1: refine GT groundings vs GT-mask bboxes ----
    report_1, log_1 = refine_report_with_mask(
        gt_report_text, mask_bboxes, min_overlap=args.refine_min_overlap)

    # ---- Reports 2/3/4 built FROM the refined report ----
    report_2, log_2 = build_report_2_ocr(
        report_1, ocr_items, iou_thresh=args.ocr_iou)
    report_3, log_3 = build_report_3_ocr_dtd(
        report_1, ocr_items, dtd_boxes, iou_thresh=args.match_iou)
    report_4, log_4, n_kept_4 = build_report_4_dtd_only(
        report_1, dtd_boxes, iou_thresh=args.match_iou)

    # ---- Report ocr_only: keep only anomalies with OCR overlap ----
    report_ocr_only, log_ocr_only, n_kept_ocr_only = build_report_ocr_only(
        report_1, ocr_items, iou_thresh=args.ocr_iou)

    # ---- Report ocr_dtd_only: keep anomalies with OCR OR DTD overlap ----
    report_ocr_dtd_only, log_ocr_dtd_only, n_kept_ocr_dtd_only = \
        build_report_ocr_dtd_only(
            report_1, ocr_items, dtd_boxes, iou_thresh=args.match_iou)

    # ---- Write everything back into the JSON ----
    payload["gt_mask_bboxes"] = [list(b) for b in mask_bboxes]
    payload["gt_mask_path"]   = str(gt_mask_path)
    payload["dtd_tp_fp"] = {
        "tp_flags":   tp_flags,
        "n_tp":       int(n_tp),
        "n_fp":       int(n_fp),
        "tp_iou":     args.tp_iou,
    }
    payload["reports"] = {
        "original":         gt_report_text,
        "report_1_refined": report_1,
        "report_2_ocr":     report_2,
        "report_3_ocr_dtd": report_3,
        "report_4_dtd_only": report_4,
        "ocr_only":         report_ocr_only,
        "ocr_dtd_only":     report_ocr_dtd_only,
    }
    payload["augmentation"] = {
        "refine_min_overlap": args.refine_min_overlap,
        "ocr_iou":            args.ocr_iou,
        "match_iou":          args.match_iou,
        "log_1_refine":       log_1,
        "log_2_ocr":          log_2,
        "log_3_ocr_dtd":      log_3,
        "log_4_dtd_only":     log_4,
        "log_ocr_only":       log_ocr_only,
        "log_ocr_dtd_only":   log_ocr_dtd_only,
        "report_4_n_kept":    n_kept_4,
        "ocr_only_n_kept":    n_kept_ocr_only,
        "ocr_dtd_only_n_kept": n_kept_ocr_dtd_only,
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Optional visualization ----
    if getattr(args, "viz_dir", None):
        try:
            image_dir = (Path(args.image_dir).expanduser().resolve()
                         if getattr(args, "image_dir", None) else None)
            img_path = _find_image(image_dir, payload.get("image_name", ""),
                                   stem)
            image_pil = Image.open(img_path).convert("RGB") if img_path else None
            viz_root = Path(args.viz_dir).expanduser().resolve()
            # mirror the partXXX structure of the json under viz_dir
            try:
                rel = json_path.parent.relative_to(
                    Path(args.dtd_ocr_dir).expanduser().resolve())
            except ValueError:
                rel = Path("")
            out_png = viz_root / rel / f"{stem}_boxes.png"
            _visualize(out_png, image_pil, payload.get("image_size"),
                       mask_bboxes=mask_bboxes, dtd_boxes=dtd_boxes,
                       tp_flags=tp_flags, ocr_items=ocr_items,
                       reports=payload["reports"],
                       draw_ocr=not args.viz_no_ocr)
            if image_pil is None:
                print(f"  [viz] {out_png.name} (no source image -> white "
                      f"canvas)", flush=True)
            else:
                print(f"  [viz] {out_png.name}", flush=True)
        except Exception as exc:
            print(f"  [viz] FAILED for {stem}: {exc!r}", flush=True)

    print(f"  [ok] {stem}  mask_bboxes={len(mask_bboxes)}  "
          f"dtd={len(dtd_boxes)} (TP={n_tp},FP={n_fp})  "
          f"refined={sum(1 for r in log_1 if r.get('replaced_with'))}/{len(log_1)}  "
          f"r4_kept={n_kept_4}  ocr_only_kept={n_kept_ocr_only}  "
          f"ocr_dtd_only_kept={n_kept_ocr_dtd_only}", flush=True)
    return "ok"


# =========================================================================== #
# Discovery
# =========================================================================== #
def discover_json(dtd_ocr_dir: Path, parts, limit: int) -> list[Path]:
    files: list[Path] = []
    if parts:
        for part in parts:
            pd = dtd_ocr_dir / part
            if not pd.is_dir():
                print(f"  [warn] part dir missing: {pd}")
                continue
            files.extend(sorted(pd.glob("*_dtd_ocr.json")))
    else:
        files = sorted(dtd_ocr_dir.rglob("*_dtd_ocr.json"))
    if limit > 0:
        files = files[:limit]
    return files


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtd_ocr_dir", required=True,
                    help="Dir of {stem}_dtd_ocr.json (extract_dtd_ocr.py out).")
    ap.add_argument("--gt_masks_dir", required=True)
    ap.add_argument("--gt_reports_dir", required=True)
    ap.add_argument("--parts", nargs="*", default=None,
                    help="Optional partXXX subdirs to process; omit for a "
                         "recursive scan of --dtd_ocr_dir.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip JSONs that already have an 'augmentation' field.")

    # Mask split-extraction (mirror extract_dtd_ocr_heatmaps.py defaults)
    ap.add_argument("--mask_min_area", type=int, default=200)
    ap.add_argument("--cc_connectivity", type=int, default=4, choices=(4, 8))
    ap.add_argument("--morph_open_ksize", type=int, default=3)
    ap.add_argument("--min_split_gap", type=int, default=2)

    # Matching thresholds
    ap.add_argument("--tp_iou", type=float, default=0.05,
                    help="Intersection-over-min for DTD TP/FP labelling.")
    ap.add_argument("--refine_min_overlap", type=float, default=0.10,
                    help="Min intersection-over-min to replace a GT grounding "
                         "with a GT-mask bbox (report 1).")
    ap.add_argument("--ocr_iou", type=float, default=0.1,
                    help="Min IoU to replace a grounding with an OCR-word "
                         "box (reports 2, 3, ocr_only and ocr_dtd_only).")
    ap.add_argument("--match_iou", type=float, default=0.1,
                    help="Min IoU for DTD-region matching (reports 3, 4 and "
                         "ocr_dtd_only).")

    # Visualization
    ap.add_argument("--viz_dir", default=None,
                    help="If set, save a per-document visualization PNG with "
                         "all box layers in distinct colours (GT-mask, DTD "
                         "TP/FP, OCR, and the four report groundings). The "
                         "partXXX structure is mirrored under this dir.")
    ap.add_argument("--image_dir", default=None,
                    help="Source images for the visualization background. If "
                         "omitted or an image is missing, boxes are drawn on a "
                         "white canvas sized to image_size from the JSON.")
    ap.add_argument("--viz_max", type=int, default=20,
                    help="If >0, only draw visualizations for the first N "
                         "documents (augmentation still runs for all).")
    ap.add_argument("--viz_no_ocr", action="store_true",
                    help="Do not draw the (dense) OCR word boxes in the "
                         "visualization.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dtd_ocr_dir   = Path(args.dtd_ocr_dir).expanduser().resolve()
    gt_masks_dir  = Path(args.gt_masks_dir).expanduser().resolve()
    gt_reports_dir = Path(args.gt_reports_dir).expanduser().resolve()
    if not dtd_ocr_dir.is_dir():
        raise SystemExit(f"--dtd_ocr_dir not a dir: {dtd_ocr_dir}")

    files = discover_json(dtd_ocr_dir, args.parts, args.limit)
    if not files:
        print("[done] no _dtd_ocr.json files found")
        return 0
    print(f"[run] {len(files)} json file(s)")

    counters = {"ok": 0, "skip_exists": 0, "no_mask": 0,
                "no_report": 0, "error": 0}
    t0 = time.time()
    n_viz = 0
    viz_dir_saved = args.viz_dir
    for i, jp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {jp.name}", flush=True)
        # Honour --viz_max: turn off viz once we've drawn N.
        if viz_dir_saved and args.viz_max > 0:
            args.viz_dir = viz_dir_saved if n_viz < args.viz_max else None
        try:
            status = process_one_json(
                jp, args=args, gt_masks_dir=gt_masks_dir,
                gt_reports_dir=gt_reports_dir)
        except Exception as exc:
            print(f"  [error] {exc!r}", flush=True)
            import traceback; traceback.print_exc()
            status = "error"
        counters[status] = counters.get(status, 0) + 1
        if status == "ok" and args.viz_dir:
            n_viz += 1

    print(f"\n[done] {counters}  elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())