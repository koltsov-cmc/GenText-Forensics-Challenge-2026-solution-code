#!/usr/bin/env python
"""Compute mIoU/mF1 for DTD probability maps across thresholds.

Evaluates three prediction modes (raw prob, naive CC bboxes, watershed bboxes)
at multiple thresholds against GT forgery masks using TruFor pixel metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# =========================================================================== #
# 1. Naive bbox extraction — copied from extract_dtd_ocr_heatmaps.py so this
#    script is self-contained. Output: [x1, y1, x2, y2] with x2 / y2 EXCLUSIVE.
# =========================================================================== #
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


def _boxes_from_submask(sub, ox, oy, min_area, min_gap=2,
                       fill_thresh=0.45, depth=0, max_depth=8):
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

    out = []
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


def _label_components(mask, connectivity):
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


def _morph_open(mask, ksize):
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


def _remove_small_objects(mask_bool: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than `min_size` pixels. Plain inline
    version to avoid skimage's deprecating `min_size` API."""
    if min_size <= 1 or not mask_bool.any():
        return mask_bool
    n, labels = _label_components(mask_bool.astype(np.uint8), 4)
    if n <= 1:
        return mask_bool
    sizes = np.bincount(labels.ravel())
    too_small = sizes < min_size
    too_small[0] = False  # never drop background
    out = mask_bool.copy()
    out[too_small[labels]] = False
    return out


def naive_regions_from_prob(prob, threshold=0.5, min_area=200,
                            connectivity=4, morph_open_ksize=3,
                            split_low_fill=True,
                            fill_split_thresh=0.45,
                            min_split_gap=2):
    """Mirrors extract_dtd_ocr_heatmaps.py:_dtd_regions_from_prob."""
    mask = (prob >= threshold).astype(np.uint8)
    mask = _morph_open(mask, morph_open_ksize)

    n, labels = _label_components(mask, connectivity)
    boxes = []
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


# =========================================================================== #
# 2. Watershed-based bbox extraction (h-maxima seeds + shallow-boundary merge).
#    Output: [x1, y1, x2, y2] with x2 / y2 EXCLUSIVE.
# =========================================================================== #
def watershed_regions_from_prob(prob, threshold=0.5,
                                peak_prominence=0.15,
                                smooth_sigma_frac=0.003,
                                min_area_frac=1e-4,
                                merge_shallow_ratio=0.85,
                                morph_open_ksize=0):
    """Marker-controlled watershed with prominence-based seeds.

    Args:
        prob:                2D float array, ideally in [0, 1].
        threshold:           foreground-mask threshold (matches `raw` mode).
        peak_prominence:     h in h-maxima — minimum bump of a peak above its
                             surroundings. Larger = fewer, more confident
                             seeds (kills noisy mini-peaks inside one blob).
        smooth_sigma_frac:   gaussian sigma as a fraction of image diagonal
                             — auto-scales with resolution.
        min_area_frac:       min region area as a fraction of image area
                             — auto-scales with resolution.
        merge_shallow_ratio: if the watershed boundary between two regions is
                             >= ratio * min(peak_a, peak_b), the split is
                             considered spurious and the regions are merged.
                             1.0 = never merge, 0.0 = merge everything.
        morph_open_ksize:    optional opening of the foreground mask (0/1
                             disables) — severs thin bridges before watershed.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed
    from skimage.measure import regionprops

    prob = np.asarray(prob, dtype=np.float32)
    H, W = prob.shape
    if prob.max() > 1.0:
        prob = prob / (prob.max() + 1e-8)

    diag = float(np.hypot(H, W))
    smooth_sigma = max(0.5, smooth_sigma_frac * diag)
    min_area = max(8, int(min_area_frac * H * W))

    smoothed = ndi.gaussian_filter(prob, sigma=smooth_sigma)
    mask = smoothed >= threshold
    if morph_open_ksize >= 2:
        mask = _morph_open(mask.astype(np.uint8),
                           morph_open_ksize).astype(bool)
    mask = _remove_small_objects(mask, min_size=min_area)
    if not mask.any():
        return []

    peaks_mask = h_maxima(smoothed * mask, h=peak_prominence)
    if not peaks_mask.any():
        # No prominent peaks — fall back to plain connected components.
        n, labels = _label_components(mask.astype(np.uint8), 4)
        out = []
        for lab in range(1, n):
            comp = (labels == lab)
            if int(comp.sum()) < min_area:
                continue
            ys, xs = np.where(comp)
            out.append([int(xs.min()), int(ys.min()),
                        int(xs.max()) + 1, int(ys.max()) + 1])
        return out

    markers, _ = ndi.label(peaks_mask)
    labels = watershed(-smoothed, markers=markers, mask=mask)
    labels = _merge_shallow_neighbors(labels, smoothed, merge_shallow_ratio)

    out = []
    for r in regionprops(labels):
        if r.area < min_area:
            continue
        y0, x0, y1, x1 = r.bbox  # half-open by skimage convention
        out.append([int(x0), int(y0), int(x1), int(y1)])
    return out


def _merge_shallow_neighbors(labels, height, ratio):
    """If the boundary between two labels is too high relative to the smaller
    of their two peaks, the watershed split is spurious — union the labels.
    Standard waterfall / region-merging post-processing for watershed."""
    from scipy import ndimage as ndi
    n = int(labels.max())
    if n < 2 or ratio >= 1.0:
        return labels

    peak_h = np.concatenate([
        [0.0],
        ndi.maximum(height, labels=labels, index=np.arange(1, n + 1)),
    ])

    pair_max: dict[tuple[int, int], float] = {}
    Hh, Ww = labels.shape
    for dy, dx in [(0, 1), (1, 0)]:
        a = labels[:Hh - dy, :Ww - dx]
        b = labels[dy:, dx:]
        ha = height[:Hh - dy, :Ww - dx]
        hb = height[dy:, dx:]
        m = (a != b) & (a > 0) & (b > 0)
        if not m.any():
            continue
        la = np.minimum(a[m], b[m]).astype(np.int64)
        lb = np.maximum(a[m], b[m]).astype(np.int64)
        bh = np.minimum(ha[m], hb[m])
        key = la * (n + 1) + lb
        order = np.argsort(key)
        key_s, bh_s = key[order], bh[order]
        uniq, starts = np.unique(key_s, return_index=True)
        ends = np.append(starts[1:], len(key_s))
        for k, s, e in zip(uniq, starts, ends):
            la_, lb_ = int(k // (n + 1)), int(k % (n + 1))
            v = float(bh_s[s:e].max())
            pair_max[(la_, lb_)] = max(pair_max.get((la_, lb_), 0.0), v)

    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (la_, lb_), border_h in pair_max.items():
        if border_h >= ratio * min(peak_h[la_], peak_h[lb_]):
            ra, rb = find(la_), find(lb_)
            if ra != rb:
                parent[ra] = rb

    remap = np.array([find(i) for i in range(n + 1)], dtype=labels.dtype)
    return remap[labels]


# =========================================================================== #
# 3. Bbox-list -> filled mask, and per-image TruFor-style pixel metric
# =========================================================================== #
def _boxes_to_mask(shape, boxes):
    """Fill [x1, y1, x2, y2] (x2/y2 exclusive) bboxes into a 0/1 mask."""
    H, W = shape
    out = np.zeros((H, W), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = 1
    return out


def _pixel_metrics(gt_b: np.ndarray, pr_b: np.ndarray):
    """Per-image pixel IoU, F1, TP/FP/FN, and no-FP metrics (TruFor-style).

    Both inputs are 0/1 uint8 of identical shape.

    Conventions:
        union == 0   (both empty)             -> IoU = F1 = 1.0
        union > 0, TP = 0                     -> IoU = F1 = 0.0
        otherwise                             -> standard formulas
    """
    inter = int(np.logical_and(gt_b, pr_b).sum())
    union = int(np.logical_or(gt_b, pr_b).sum())
    if union == 0:
        iou = 1.0
    else:
        iou = inter / union

    tp = inter
    fp = int(np.logical_and(pr_b == 1, gt_b == 0).sum())
    fn = int(np.logical_and(pr_b == 0, gt_b == 1).sum())

    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0

    # no-FP: hypothetical ceiling where every predicted pixel is correct
    if tp == 0 and fn == 0:
        iou_no_fp = 1.0
        f1_no_fp = 1.0
    else:
        union_no_fp = tp + fn
        iou_no_fp = tp / union_no_fp if union_no_fp > 0 else 0.0
        denom_no_fp = 2 * tp + fn
        f1_no_fp = (2 * tp) / denom_no_fp if denom_no_fp > 0 else 0.0

    return float(iou), float(f1), tp, fp, fn, float(iou_no_fp), float(f1_no_fp)


# =========================================================================== #
# 4. Pair discovery (unchanged from original script)
# =========================================================================== #
def _find_paired(probs_dir: Path, masks_dir: Path):
    probs = {}
    for p in sorted(probs_dir.rglob("*_dtd_prob.npy")):
        stem = p.name.replace("_dtd_prob.npy", "")
        probs[stem] = p
    
    if len(probs) == 0:
        for p in sorted(probs_dir.rglob("*.dtd.prob.npy")):
            stem = p.name.replace(".dtd.prob.npy", "")
            probs[stem] = p

    masks = {}
    for m in sorted(masks_dir.rglob("*")):
        if not m.is_file():
            continue
        if m.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            stem = m.stem
            if stem.endswith("_mask"):
                stem = stem[:-5]
            masks[stem] = m

    pairs = []
    n_missing = 0
    for stem, prob_path in sorted(probs.items()):
        if stem in masks:
            pairs.append((prob_path, masks[stem]))
        else:
            pairs.append((prob_path, None))
            n_missing += 1
    print(f"[data] {len(pairs)} samples  "
          f"({len(probs)} probs, {len(masks)} masks, "
          f"{n_missing} pristine)")
    return pairs


# =========================================================================== #
# 5. Main: load each prob ONCE, evaluate all modes x all thresholds
# =========================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--probs_dir", required=True)
    ap.add_argument("--gt_masks_dir", required=True)
    ap.add_argument(
        "--thresholds",
        default="0.2,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.80,0.85,0.90,0.95,0.98,0.99",
        help="Comma-separated thresholds to test.",
    )
    ap.add_argument(
        "--modes", default="naive",
        help="Subset of {raw, naive, watershed}, comma-separated.",
    )
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, evaluate on the first N pairs only.")

    # Naive params — same defaults as extract_dtd_ocr_heatmaps.py CLI
    g_n = ap.add_argument_group("naive bbox extraction")
    g_n.add_argument("--naive_min_area", type=int, default=200)
    g_n.add_argument("--naive_connectivity", type=int, default=4,
                     choices=(4, 8))
    g_n.add_argument("--naive_morph_open_ksize", type=int, default=3)
    g_n.add_argument("--naive_no_split_low_fill", action="store_true")
    g_n.add_argument("--naive_fill_split_thresh", type=float, default=0.45)
    g_n.add_argument("--naive_min_split_gap", type=int, default=2)

    # Watershed params
    g_w = ap.add_argument_group("watershed bbox extraction")
    g_w.add_argument("--ws_peak_prominence",   type=float, default=0.15,
                     help="h in h-maxima; larger -> fewer / stronger peaks.")
    g_w.add_argument("--ws_smooth_sigma_frac", type=float, default=0.003,
                     help="Smoothing sigma as fraction of image diagonal.")
    g_w.add_argument("--ws_min_area_frac",     type=float, default=1e-4,
                     help="Min region area as fraction of image area.")
    g_w.add_argument("--ws_merge_shallow_ratio", type=float, default=0.85,
                     help="Merge soft-split neighbours (1.0 disables).")
    g_w.add_argument("--ws_morph_open_ksize",  type=int, default=0,
                     help="Optional opening before watershed (0/1 disables).")

    args = ap.parse_args()

    probs_dir = Path(args.probs_dir).expanduser().resolve()
    masks_dir = Path(args.gt_masks_dir).expanduser().resolve()
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("raw", "naive", "watershed"):
            raise SystemExit(f"unknown mode: {m}")

    pairs = _find_paired(probs_dir, masks_dir)
    if args.limit > 0:
        pairs = pairs[:args.limit]
        print(f"[data] limited to first {len(pairs)} pairs")
    if not pairs:
        print("[error] no paired samples found")
        return 1

    # accum[mode][thr] = [sum_iou, sum_f1, sum_tp, sum_fp, sum_fn,
    #                      sum_iou_no_fp, sum_f1_no_fp, n]
    accum = {m: {thr: [0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0] for thr in thresholds}
             for m in modes}

    for prob_path, mask_path in tqdm(pairs, desc="images"):
        prob = np.load(prob_path).astype(np.float32)
        if prob.ndim != 2:
            prob = prob.squeeze()
        H, W = prob.shape

        if mask_path is None:
            gt = np.zeros((H, W), dtype=np.uint8)
        else:
            gt_img = np.array(Image.open(mask_path).convert("L"),
                              dtype=np.uint8)
            if gt_img.shape != (H, W):
                # Resize GT to prob resolution (NEAREST keeps it binary).
                gt_img = np.array(
                    Image.fromarray(gt_img).resize((W, H), Image.NEAREST),
                    dtype=np.uint8,
                )
            gt = (gt_img > 0).astype(np.uint8)

        for thr in thresholds:
            if "raw" in modes:
                pred = (prob >= thr).astype(np.uint8)
                iou, f1, tp, fp, fn, iou_nf, f1_nf = _pixel_metrics(gt, pred)
                s = accum["raw"][thr]
                s[0] += iou; s[1] += f1; s[2] += tp; s[3] += fp; s[4] += fn
                s[5] += iou_nf; s[6] += f1_nf; s[7] += 1

            if "naive" in modes:
                boxes = naive_regions_from_prob(
                    prob, threshold=thr,
                    min_area=args.naive_min_area,
                    connectivity=args.naive_connectivity,
                    morph_open_ksize=args.naive_morph_open_ksize,
                    split_low_fill=not args.naive_no_split_low_fill,
                    fill_split_thresh=args.naive_fill_split_thresh,
                    min_split_gap=args.naive_min_split_gap,
                )
                pred = _boxes_to_mask((H, W), boxes)
                iou, f1, tp, fp, fn, iou_nf, f1_nf = _pixel_metrics(gt, pred)
                s = accum["naive"][thr]
                s[0] += iou; s[1] += f1; s[2] += tp; s[3] += fp; s[4] += fn
                s[5] += iou_nf; s[6] += f1_nf; s[7] += 1

            if "watershed" in modes:
                boxes = watershed_regions_from_prob(
                    prob, threshold=thr,
                    peak_prominence=args.ws_peak_prominence,
                    smooth_sigma_frac=args.ws_smooth_sigma_frac,
                    min_area_frac=args.ws_min_area_frac,
                    merge_shallow_ratio=args.ws_merge_shallow_ratio,
                    morph_open_ksize=args.ws_morph_open_ksize,
                )
                pred = _boxes_to_mask((H, W), boxes)
                iou, f1, tp, fp, fn, iou_nf, f1_nf = _pixel_metrics(gt, pred)
                s = accum["watershed"][thr]
                s[0] += iou; s[1] += f1; s[2] += tp; s[3] += fp; s[4] += fn
                s[5] += iou_nf; s[6] += f1_nf; s[7] += 1

    # -------- pretty-print table ---------------------------------------- #
    col_w = 9
    header = f"{'Thr':>5}"
    for m in modes:
        header += (f" | {m+'_IoU':>{col_w}} {m+'_F1':>{col_w}} "
                   f"{m+'_FS':>{col_w}} {m+'_IoUnf':>{col_w}} "
                   f"{m+'_F1nf':>{col_w}} {m+'_FSnf':>{col_w}}")
    print()
    print(header)
    print("-" * len(header))
    for thr in thresholds:
        row = f"{thr:5.2f}"
        for m in modes:
            s = accum[m][thr]
            n = s[7]
            mi = s[0] / max(1, n)
            mf = s[1] / max(1, n)
            fs = 0.5 * mi + 0.5 * mf
            mi_nf = s[5] / max(1, n)
            mf_nf = s[6] / max(1, n)
            fs_nf = 0.5 * mi_nf + 0.5 * mf_nf
            row += (f" | {mi:{col_w}.4f} {mf:{col_w}.4f} {fs:{col_w}.4f} "
                    f"{mi_nf:{col_w}.4f} {mf_nf:{col_w}.4f} {fs_nf:{col_w}.4f}")
        print(row)

    # -------- per-mode best -------------------------------------------- #
    print("\n[best per mode]")
    for m in modes:
        best_iou = max(thresholds,
                       key=lambda t: accum[m][t][0] / max(1, accum[m][t][7]))
        best_f1 = max(thresholds,
                      key=lambda t: accum[m][t][1] / max(1, accum[m][t][7]))
        best_fs = max(thresholds,
                      key=lambda t: (
                          0.5 * accum[m][t][0] / max(1, accum[m][t][7]) +
                          0.5 * accum[m][t][1] / max(1, accum[m][t][7])))
        mi = accum[m][best_iou][0] / max(1, accum[m][best_iou][7])
        mf = accum[m][best_f1][1] / max(1, accum[m][best_f1][7])
        fs = 0.5 * mi + 0.5 * mf

        best_iou_nf = max(thresholds,
                          key=lambda t: accum[m][t][5] / max(1, accum[m][t][7]))
        best_f1_nf = max(thresholds,
                         key=lambda t: accum[m][t][6] / max(1, accum[m][t][7]))
        best_fs_nf = max(thresholds,
                         key=lambda t: (
                             0.5 * accum[m][t][5] / max(1, accum[m][t][7]) +
                             0.5 * accum[m][t][6] / max(1, accum[m][t][7])))
        mi_nf = accum[m][best_iou_nf][5] / max(1, accum[m][best_iou_nf][7])
        mf_nf = accum[m][best_f1_nf][6] / max(1, accum[m][best_f1_nf][7])
        fs_nf = 0.5 * mi_nf + 0.5 * mf_nf

        print(f"  {m:>10}: best mIoU={mi:.4f} @ thr={best_iou:.2f}   "
              f"best mF1={mf:.4f} @ thr={best_f1:.2f}   "
              f"best FS={fs:.4f} @ thr={best_fs:.2f}")
        print(f"              no-FP  mIoU={mi_nf:.4f} @ thr={best_iou_nf:.2f}   "
              f"mF1={mf_nf:.4f} @ thr={best_f1_nf:.2f}   "
              f"FS={fs_nf:.4f} @ thr={best_fs_nf:.2f}")

    # -------- per-threshold pixel stats -------------------------------- #
    print("\n[per-threshold pixel stats]")
    for m in modes:
        print(f"\n  --- {m} ---")
        print(f"  {'Thr':>5} | {'avg_TP':>10} {'avg_FP':>10} {'avg_FN':>10} | "
              f"{'total_TP':>12} {'total_FP':>12} {'total_FN':>12} | "
              f"{'FS':>8} {'FSnf':>8}")
        print("  " + "-" * 90)
        for thr in thresholds:
            s = accum[m][thr]
            n = s[7]
            avg_tp = s[2] / max(1, n)
            avg_fp = s[3] / max(1, n)
            avg_fn = s[4] / max(1, n)
            mi = s[0] / max(1, n)
            mf = s[1] / max(1, n)
            fs = 0.5 * mi + 0.5 * mf
            mi_nf = s[5] / max(1, n)
            mf_nf = s[6] / max(1, n)
            fs_nf = 0.5 * mi_nf + 0.5 * mf_nf
            print(
                f"  {thr:5.2f} | {avg_tp:10.1f} {avg_fp:10.1f} {avg_fn:10.1f} | "
                f"{s[2]:12,} {s[3]:12,} {s[4]:12,} | "
                f"{fs:8.4f} {fs_nf:8.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())