#!/usr/bin/env python
"""PaddleOCR wrapper with Sobel-based word splitting + auto language detection.

Iterates candidate language codes, picks the best via length-weighted confidence.
Splits multi-word blocks using gradient-magnitude projection (skips zh/th).
Provides both word-level items and sentence-level reading_order_text.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Language config
# --------------------------------------------------------------------------- #
# Languages that use whitespace between words and read left-to-right.
_LTR_SPACED_LANGS = {"en", "ms", "id", "fr", "de", "es", "it", "pt", "ru", "uk"}

# Right-to-left scripts (still space-separated between words).
_RTL_SPACED_LANGS = {"ar", "fa", "ur", "he"}

# Scriptio continua: little or no inter-word whitespace.
# Splitting on whitespace is meaningless here.
_SCRIPTIO_CONTINUA_LANGS = {"zh", "ch", "ch_tra", "th", "ja", "ko"}


def _is_space_separated(lang: str) -> bool:
    return lang in _LTR_SPACED_LANGS or lang in _RTL_SPACED_LANGS


def _is_rtl(lang: str) -> bool:
    return lang in _RTL_SPACED_LANGS


# --------------------------------------------------------------------------- #
# Geometry helpers (kept from the original script)
# --------------------------------------------------------------------------- #
def _poly_to_bbox(poly: Sequence[Sequence[float]]) -> List[int]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _group_into_lines(items: List[dict], y_tolerance: float = 0.35) -> List[List[dict]]:
    if not items:
        return []
    heights = [it["bbox"][3] - it["bbox"][1] for it in items]
    med_h = float(np.median(heights)) if heights else 20.0
    threshold = med_h * y_tolerance
    sorted_items = sorted(items, key=lambda it: it["bbox"][1])
    lines: List[List[dict]] = []
    for it in sorted_items:
        y1, y2 = it["bbox"][1], it["bbox"][3]
        placed = False
        for line in lines:
            line_y1 = min(i["bbox"][1] for i in line)
            line_y2 = max(i["bbox"][3] for i in line)
            overlap = min(y2, line_y2) - max(y1, line_y1)
            if overlap >= -threshold:
                line.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])
    for line in lines:
        line.sort(key=lambda it: it["bbox"][0])
    return lines


def _is_sentence(text: str) -> bool:
    """A 'sentence' here = a block with 3+ space-separated tokens."""
    return len(text.split()) >= 3 or text.count(" ") >= 2


# --------------------------------------------------------------------------- #
# Sobel-based word splitter
# --------------------------------------------------------------------------- #
def _smooth_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Tiny 1-D Gaussian smoothing without bringing in scipy."""
    if sigma <= 0:
        return arr
    radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return [(start, end)] (end exclusive) for each contiguous True run."""
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_run:
            start = i
            in_run = True
        elif not m and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def _proportional_fallback(
    words: List[str],
    sentence_bbox: List[int],
    rtl: bool,
    confidence: float,
) -> List[dict]:
    """Equal-width fallback when Sobel can't find enough gaps."""
    x1, y1, x2, y2 = sentence_bbox
    W = max(1, x2 - x1)
    weights = [max(1, len(w)) for w in words]
    total = float(sum(weights))

    iter_words = list(reversed(words)) if rtl else list(words)
    iter_weights = list(reversed(weights)) if rtl else list(weights)

    items: List[dict] = []
    cur = float(x1)
    for w, wt in zip(iter_words, iter_weights):
        ww = W * wt / total
        items.append({
            "text": w,
            "bbox": [int(round(cur)), y1, int(round(cur + ww)), y2],
            "confidence": confidence,
        })
        cur += ww
    if rtl:
        items = list(reversed(items))  # restore logical reading order
    return items


def _split_sentence_with_sobel(
    image_array: np.ndarray,
    sentence_bbox: List[int],
    text: str,
    *,
    rtl: bool = False,
    threshold_ratio: float = 0.06,
    smoothing_factor: float = 0.05,  # sigma_x = H_tight * smoothing_factor
    confidence: float = 1.0,
    debug_dir: str | Path | None = None,
    debug_id: str = "",
) -> List[dict]:
    """Split a sentence-level bbox into word-level bboxes via Sobel projection.

    Algorithm (improved):
        1. Crop the sentence region.
        2. Compute the gradient magnitude  |G| = sqrt(Gx**2 + Gy**2).
           Using the full magnitude (not just |Sobel-X|) avoids false gaps
           at columns that contain only horizontal strokes — middle bars
           of `e`/`s`, the connecting curves of `n`/`u`/`m`, em-dashes,
           Cyrillic crossbars, etc. A column is "empty" only if it has
           no edge response in EITHER axis.
        3. Vertically tighten — find the densest contiguous Y-band of ink
           (drops ascenders/descenders from neighbouring lines).
        4. Recompute |G| on the tight crop and project onto the X axis.
        5. Horizontally tighten — find the leftmost/rightmost ink columns
           (strips PaddleOCR bbox padding so word slots line up with text).
        6. Find all "ink-empty" runs inside [x_left, x_right].
        7. Decide which runs are inter-word vs inter-letter via bimodal
           clustering (largest jump in sorted gap-width distribution),
           with "N widest" as fallback if clustering doesn't give the
           expected count.
        8. Map runs to word bboxes; for RTL scripts the leftmost slot
           corresponds to the LAST word in logical text order.
    """
    words = text.split()
    n_words = len(words)
    if n_words <= 1:
        return [{
            "text": text,
            "bbox": list(sentence_bbox),
            "confidence": confidence,
        }]

    x1, y1, x2, y2 = sentence_bbox
    H_full, W_full = image_array.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W_full, x2), min(H_full, y2)
    crop = image_array[y1c:y2c, x1c:x2c]

    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return _proportional_fallback(words, sentence_bbox, rtl, confidence)

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop
    H_orig, W_orig = gray.shape

    # ---- (1) Gradient magnitude on the full crop ----
    # Compute both Sobel-X (vertical strokes) and Sobel-Y (horizontal strokes)
    # and combine them with L2 magnitude. A pixel is "empty" only if neither
    # axis has any edge response — this is what we want for finding true
    # background between words.
    gx_full = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_full = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(gx_full, gy_full)  # already non-negative

    # ---- (2) Vertical tightening: largest dense Y-band ----
    row_profile = sobel_mag.sum(axis=1).astype(np.float32)
    row_profile_s = _smooth_1d(row_profile, sigma=max(1.0, H_orig * 0.03))
    row_thresh = max(row_profile_s.max() * 0.15, 1e-3)
    bands = _find_runs(row_profile_s > row_thresh)
    if bands:
        y_top_l, y_bot_l = max(bands, key=lambda b: b[1] - b[0])
        margin_y = max(1, int(0.05 * (y_bot_l - y_top_l)))
        y_top_l = max(0, y_top_l - margin_y)
        y_bot_l = min(H_orig, y_bot_l + margin_y)
    else:
        y_top_l, y_bot_l = 0, H_orig
    if y_bot_l - y_top_l < 3:
        return _proportional_fallback(words, sentence_bbox, rtl, confidence)

    gray_tight = gray[y_top_l:y_bot_l]
    H_tight = gray_tight.shape[0]

    # ---- (3) X projection of |G| on the tightened crop ----
    gx_t = cv2.Sobel(gray_tight, cv2.CV_32F, 1, 0, ksize=3)
    gy_t = cv2.Sobel(gray_tight, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_tight = cv2.magnitude(gx_t, gy_t)
    col_profile = grad_mag_tight.sum(axis=0).astype(np.float32)
    col_profile_s = _smooth_1d(
        col_profile, sigma=max(1.0, H_tight * smoothing_factor)
    )

    p_min = float(col_profile_s.min())
    p_max = float(col_profile_s.max())
    rng = p_max - p_min
    if rng <= 1e-6:
        return _proportional_fallback(words, sentence_bbox, rtl, confidence)
    profile_norm = (col_profile_s - p_min) / rng

    # ---- (4) Horizontal tightening: trim padding columns ----
    has_ink = profile_norm > threshold_ratio
    if not has_ink.any():
        return _proportional_fallback(words, sentence_bbox, rtl, confidence)
    ink_cols = np.where(has_ink)[0]
    x_left = int(ink_cols[0])
    x_right = int(ink_cols[-1]) + 1
    if x_right - x_left < 3:
        return _proportional_fallback(words, sentence_bbox, rtl, confidence)

    # ---- (5) Find gaps INSIDE the inked region only ----
    inked_profile = profile_norm[x_left:x_right]
    is_gap_local = inked_profile < threshold_ratio
    runs_local = _find_runs(is_gap_local)
    interior_runs = [
        (s, e) for s, e in runs_local
        if s > 0 and e < len(inked_profile)
    ]

    expected_gaps = n_words - 1

    # ---- (6) Choose word-separator gaps ----
    word_gaps: List[Tuple[int, int]] = []
    if expected_gaps == 0:
        word_gaps = []
    elif len(interior_runs) < expected_gaps:
        return _proportional_fallback(words, sentence_bbox, rtl, confidence)
    elif len(interior_runs) == expected_gaps:
        word_gaps = sorted(interior_runs)
    else:
        # Bimodal clustering: find the biggest jump in sorted gap widths.
        widths = sorted([e - s for s, e in interior_runs])
        diffs = np.diff(widths)
        if len(diffs) > 0 and diffs.max() > 0:
            jump_idx = int(np.argmax(diffs))
            cut_w = (widths[jump_idx] + widths[jump_idx + 1]) / 2.0
            wide = [(s, e) for s, e in interior_runs if (e - s) > cut_w]
            if len(wide) == expected_gaps:
                word_gaps = sorted(wide)
        if not word_gaps:
            # Fallback: just take the N widest gaps.
            widest = sorted(
                interior_runs, key=lambda r: -(r[1] - r[0])
            )[:expected_gaps]
            word_gaps = sorted(widest)

    # ---- (7) Build word bboxes using TIGHTENED extents ----
    splits_local = [(s + e) // 2 for s, e in word_gaps]
    boundaries_local = [0] + splits_local + [x_right - x_left]

    word_order = list(reversed(words)) if rtl else list(words)

    word_items: List[dict] = []
    for i, w in enumerate(word_order):
        wx1 = x1c + x_left + boundaries_local[i]
        wx2 = x1c + x_left + boundaries_local[i + 1]
        wy1 = y1c + y_top_l
        wy2 = y1c + y_bot_l
        word_items.append({
            "text": w,
            "bbox": [int(wx1), int(wy1), int(wx2), int(wy2)],
            "confidence": confidence,
        })

    if rtl:
        word_items = list(reversed(word_items))

    if debug_dir:
        _save_debug_plot(
            crop, profile_norm, word_gaps, x_left, x_right,
            y_top_l, y_bot_l, words, debug_dir, debug_id,
        )

    return word_items


def _save_debug_plot(
    crop: np.ndarray,
    profile_norm: np.ndarray,
    word_gaps: List[Tuple[int, int]],
    x_left: int,
    x_right: int,
    y_top: int,
    y_bot: int,
    words: List[str],
    debug_dir: str | Path,
    debug_id: str,
) -> None:
    """Save a per-sentence debug plot: crop with overlays + Sobel profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 5),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    axes[0].imshow(crop)
    axes[0].axhline(y_top, color="cyan", linewidth=1, alpha=0.7)
    axes[0].axhline(y_bot, color="cyan", linewidth=1, alpha=0.7)
    axes[0].axvline(x_left, color="red", linewidth=1, alpha=0.7)
    axes[0].axvline(x_right, color="red", linewidth=1, alpha=0.7)
    for s, e in word_gaps:
        axes[0].axvspan(x_left + s, x_left + e,
                        color="orange", alpha=0.4)
    axes[0].set_title(
        f"crop  |  expected words: {' | '.join(words)}  "
        f"|  red=x_tight  cyan=y_tight  orange=word-gap"
    )
    axes[0].axis("off")

    axes[1].plot(profile_norm, linewidth=0.8)
    axes[1].axvline(x_left, color="red", linewidth=1, alpha=0.7)
    axes[1].axvline(x_right, color="red", linewidth=1, alpha=0.7)
    for s, e in word_gaps:
        axes[1].axvspan(x_left + s, x_left + e,
                        color="orange", alpha=0.4)
    axes[1].set_title("normalised |G| = sqrt(Gx² + Gy²) column projection")
    axes[1].set_xlim(0, len(profile_norm))
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    out = debug_dir / f"debug_{debug_id}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Reusable OCR engine cache
# --------------------------------------------------------------------------- #
class _PaddleOcrEngine:
    """Lazy-initialising per-language PaddleOCR cache."""

    def __init__(self, gpu: bool, gpu_id: int = 0):
        self.gpu = gpu
        self.gpu_id = gpu_id
        self._engines: dict[str, any] = {}

    def get(self, lang: str):
        from paddleocr import PaddleOCR
        if lang not in self._engines:
            device = f"gpu:{self.gpu_id}" if self.gpu else "cpu"
            self._engines[lang] = PaddleOCR(
                lang=lang,
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                return_word_box=False,
            )
        return self._engines[lang]


# --------------------------------------------------------------------------- #
# Single-language OCR pass
# --------------------------------------------------------------------------- #
def _run_paddle_pass(
    pil: Image.Image,
    *,
    lang: str,
    engine: _PaddleOcrEngine,
    mag_ratio: float,
) -> Tuple[List[dict], float]:
    """Run one PaddleOCR pass for a given language using a cached engine.

    Returns
    -------
    (sentence_items, weighted_mean_conf)
        sentence_items: list of {text, bbox, confidence} in original
                        image coordinates.
        weighted_mean_conf: per-character mean confidence — the
                        language-detection score.
    """
    ocr = engine.get(lang)

    orig_w, orig_h = pil.size
    if mag_ratio != 1.0:
        pil_for_ocr = pil.resize(
            (int(round(orig_w * mag_ratio)), int(round(orig_h * mag_ratio))),
            Image.BILINEAR,
        )
    else:
        pil_for_ocr = pil
    img_array = np.array(pil_for_ocr)

    texts: List[str] = []
    scores: List[float] = []
    boxes: List = []

    # 1) Modern predict() API
    try:
        result = ocr.predict(img_array)
        res = next(iter(result))
        payload = getattr(res, "json", {})
        payload = payload.get("res", payload)
        texts = list(payload.get("rec_texts", []))
        scores = list(payload.get("rec_scores", []))
        boxes = list(payload.get("rec_boxes", []))
    except Exception:
        # 2) Legacy ocr() API fallback
        raw = ocr.ocr(img_array)
        if raw and raw[0]:
            for item in raw[0]:
                if isinstance(item, (list, tuple)):
                    if len(item) == 3:
                        bbox_poly, text, conf = item
                    elif len(item) == 2 and isinstance(item[1], (list, tuple)):
                        bbox_poly, (text, conf) = item
                    else:
                        continue
                else:
                    continue
                boxes.append(bbox_poly)
                texts.append(str(text))
                scores.append(float(conf))

    items: List[dict] = []
    for text, conf, box in zip(texts, scores, boxes):
        if isinstance(box[0], (list, tuple, np.ndarray)):
            x1, y1, x2, y2 = _poly_to_bbox(box)
        else:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
        if mag_ratio != 1.0:
            x1, y1, x2, y2 = [int(round(v / mag_ratio)) for v in (x1, y1, x2, y2)]
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        items.append({
            "text": str(text),
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
        })

    # Length-weighted mean confidence: prevents a single high-conf token
    # from beating a long, well-recognised passage.
    if not items:
        score = 0.0
    else:
        total_chars = sum(max(1, len(it["text"])) for it in items)
        score = sum(it["confidence"] * max(1, len(it["text"]))
                    for it in items) / total_chars

    return items, float(score)


# --------------------------------------------------------------------------- #
# Top-level orchestrator: language detection + word splitting
# --------------------------------------------------------------------------- #
def run_paddle_ocr_with_lang_detect(
    image_path: str | Path,
    *,
    candidate_langs: Sequence[str] = ("en", "zh", "th", "ms", "id", "ar"),
    gpu: bool = True,
    mag_ratio: float = 2.0,
    y_tolerance: float = 0.35,
    gpu_id: int = 0,
    verbose: bool = True,
    debug_dir: str | Path | None = None,
    confidence_threshold: float = 0.90,
    engine: _PaddleOcrEngine | None = None,
) -> dict:
    """Detect language and run word-level OCR with Sobel-based splitting.

    Stops after the first language whose weighted confidence exceeds
    *confidence_threshold* (default 0.90).  Falls back to the best-scoring
    language if none crosses the threshold.
    """
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    pil = Image.open(str(image_path)).convert("RGB")
    image_array = np.array(pil)

    if engine is None:
        engine = _PaddleOcrEngine(gpu=gpu, gpu_id=gpu_id)

    # ---------- language detection: one full PaddleOCR pass per candidate ----
    lang_results: dict[str, dict] = {}
    best_lang: str | None = None
    for lang in candidate_langs:
        if verbose:
            print(f"[lang-detect] trying lang={lang} ...", flush=True)
        t0 = time.time()
        try:
            items, score = _run_paddle_pass(
                pil, lang=lang, engine=engine, mag_ratio=mag_ratio,
            )
        except Exception as exc:
            print(f"[lang-detect]   {lang}: FAILED — {exc!r}", flush=True)
            lang_results[lang] = {
                "items": [], "score": 0.0, "n_items": 0, "elapsed": 0.0,
            }
            continue
        elapsed = time.time() - t0
        lang_results[lang] = {
            "items": items,
            "score": score,
            "n_items": len(items),
            "elapsed": elapsed,
        }
        if verbose:
            print(
                f"[lang-detect]   {lang}: items={len(items):>3d}  "
                f"weighted_conf={score:.4f}  ({elapsed:.1f}s)",
                flush=True,
            )
        if score >= confidence_threshold:
            best_lang = lang
            if verbose:
                print(
                    f"[lang-detect]   >>> early stop: {lang} "
                    f"crossed threshold {confidence_threshold} <<<")
            break

    if not lang_results or all(r["score"] == 0.0 for r in lang_results.values()):
        raise RuntimeError("No language pass produced any text.")

    if best_lang is None:
        best_lang = max(
            lang_results.keys(),
            key=lambda L: (lang_results[L]["score"], lang_results[L]["n_items"]),
        )
    if verbose:
        print(f"\n[lang-detect] >>> selected lang={best_lang} <<<", flush=True)

    sentence_items = lang_results[best_lang]["items"]
    rtl = _is_rtl(best_lang)
    do_split = _is_space_separated(best_lang)

    # ---------- word splitting via Sobel ------------------------------------
    ocr_items: List[dict] = []
    for s_idx, item in enumerate(sentence_items):
        if do_split and _is_sentence(item["text"]):
            word_items = _split_sentence_with_sobel(
                image_array, item["bbox"], item["text"],
                rtl=rtl, confidence=item["confidence"],
                debug_dir=debug_dir,
                debug_id=f"{s_idx:03d}",
            )
            ocr_items.extend(word_items if word_items else [item])
        else:
            ocr_items.append(item)

    # reading_order_text stays at sentence level (cleaner for downstream LMs).
    lines = _group_into_lines(sentence_items, y_tolerance=y_tolerance)
    reading_parts = [" ".join(it["text"] for it in line) for line in lines]
    reading_order_text = "\n".join(reading_parts)

    for i, it in enumerate(ocr_items, start=1):
        it["id"] = i

    return {
        "reading_order_text": reading_order_text,
        "ocr_items": ocr_items,
        "image": str(image_path),
        "lang": best_lang,
        "language_scores": {L: r["score"] for L, r in lang_results.items()},
        "language_item_counts": {L: r["n_items"] for L, r in lang_results.items()},
        "language_timings_s": {L: r["elapsed"] for L, r in lang_results.items()},
        "n_items": len(ocr_items),
        "n_sentences": len(sentence_items),
    }


# --------------------------------------------------------------------------- #
# Formatting / viz / CLI
# --------------------------------------------------------------------------- #
def format_for_prompt(ocr_result: dict) -> str:
    lines = [
        "--- OCR EXTRACTION ---",
        f"(detected language: {ocr_result.get('lang', 'unknown')})",
    ]
    if "language_scores" in ocr_result:
        scores_str = ", ".join(
            f"{L}={s:.3f}" for L, s in ocr_result["language_scores"].items()
        )
        lines.append(f"(language scores: {scores_str})")
    lines.extend([
        "",
        "Reading-order text:",
        "",
        ocr_result["reading_order_text"],
        "",
        "Individual word/text blocks with coordinates:",
    ])
    for it in ocr_result["ocr_items"]:
        x1, y1, x2, y2 = it["bbox"]
        lines.append(
            f'  [{it["id"]:>3d}]  [{x1:>4d},{y1:>4d},{x2:>4d},{y2:>4d}]  '
            f'(conf={it["confidence"]:.2f})  "{it["text"]}"'
        )
    lines.append("--- END OCR ---")
    return "\n".join(lines)


def draw_ocr_boxes(
    image_path: str | Path,
    ocr_items: list[dict],
    out_path: str | Path,
    title: str = "",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    image = Image.open(str(image_path)).convert("RGB")
    w, h = image.size
    fig_w = min(18, max(8, w / 80))
    fig_h = max(6, fig_w * h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)

    for it in ocr_items:
        x1, y1, x2, y2 = it["bbox"]
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        rect = mpatches.Rectangle(
            (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
            linewidth=1.5, edgecolor="#00c853", facecolor="none",
        )
        ax.add_patch(rect)
        label = f"[{it['id']}] {it['text'][:20]}"
        ax.text(
            x1, max(0, y1 - 4), label, color="white", fontsize=7,
            bbox=dict(facecolor="#00c853", edgecolor="none", alpha=0.85, pad=1.5),
        )

    fig.tight_layout()
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ocr] viz saved to {out}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image", required=True, help="Path to input image.")
    ap.add_argument(
        "--langs",
        default="en,zh,th,ms,id,ar",
        help="Comma-separated PaddleOCR language codes to try (default: en,zh,th,ms,id,ar).",
    )
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--mag_ratio", type=float, default=1.0)
    ap.add_argument("--out_json", help="Save full JSON result.")
    ap.add_argument("--out_txt",  help="Save formatted prompt text.")
    ap.add_argument("--viz",      help="Save bbox visualization PNG.")
    ap.add_argument(
        "--debug_dir",
        help="If set, dump per-sentence Sobel-split debug PNGs here.",
    )
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-language progress lines.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    gpu = False if args.cpu else args.gpu
    candidate_langs = [s.strip() for s in args.langs.split(",") if s.strip()]

    result = run_paddle_ocr_with_lang_detect(
        args.image,
        candidate_langs=candidate_langs,
        gpu=gpu,
        mag_ratio=args.mag_ratio,
        verbose=not args.quiet,
        debug_dir=args.debug_dir,
    )

    print(
        f"\n[ocr] detected lang={result['lang']}  "
        f"sentences={result['n_sentences']}  word_items={result['n_items']}"
    )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ocr] JSON -> {out}")

    prompt_text = format_for_prompt(result)
    if args.out_txt:
        out = Path(args.out_txt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(prompt_text, encoding="utf-8")
        print(f"[ocr] text -> {out}")

    if args.viz:
        draw_ocr_boxes(
            args.image, result["ocr_items"], args.viz,
            title=(f"OCR sobel-split  lang={result['lang']}  "
                   f"items={result['n_items']}"),
        )

    print("\n--- PROMPT TEXT ---\n")
    print(prompt_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())    