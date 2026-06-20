#!/usr/bin/env python
"""Inference for trained student VLM — strict parity with train_student.py.

Two regimes:
  --regime_all         Single pass (one LoRA adapter or zero-shot).
  --regime_two_stage   Two passes: Stage 1 filters DTD anomalies, Stage 2
                       generates the full forensic report with OCR.

Prompt sources: --prompt_template (live OCR/DTD rendering) or --prompt_folder
(pre-rendered .prompt.txt files). Supports pre-rendered heatmaps via --heatmap_dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from transformers import TextStreamer


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
from realtext_v2.report import parse_report, ForgeryReport  # noqa: E402

sys.path.insert(0, str(_TOOLKIT_ROOT / "scripts"))
from run_paddle_sobel import (                                          # noqa: E402
    run_paddle_ocr_with_lang_detect, draw_ocr_boxes,
)
try:
    from vis_report import visualize_report, visualize_report_with_mask  # noqa: E402
    _HAS_VIS_REPORT = True
except ImportError:
    _HAS_VIS_REPORT = False
    print("[warn] vis_report not found in scripts/ — pred_viz/gt_viz skipped")

_DTD_SCRIPT_DIR = _TOOLKIT_ROOT / "ForensicHub" / "dtd_train"


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_REPORT_ANCHOR_RE = re.compile(r"#\s*FORGERY\s+ANALYSIS\s+REPORT", re.IGNORECASE)
_END_MARKER = "**END OF REPORT**"


# --------------------------------------------------------------------------- #
# OCR / DTD helpers — verbatim parity with prerender_prompts.py
# --------------------------------------------------------------------------- #
def _bbox_overlap_frac(box_a, box_b) -> float:
    """Fraction of box_b's area covered by box_a. 0.0 if no overlap."""
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


def _dtd_regions_from_prob(prob: np.ndarray, threshold: float = 0.98,
                            min_area: int = 200) -> list[list[int]]:
    mask = (prob >= threshold).astype(np.uint8) * 255
    return [list(b) for b in mask_to_boxes(mask, min_area=min_area)]


def _format_dtd_hints_with_overlap(
    dtd_regions: list,
    ocr_items: list,
    min_overlap_frac: float = 0.4,
    max_overlapping_items: int = 6,
) -> str:
    if not dtd_regions:
        return "No suspicious regions detected by DTD."
    lines = [f"DTD flagged {len(dtd_regions)} suspicious region(s):"]
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
                overlaps.append((frac, it))
        overlaps.sort(key=lambda kv: -kv[0])
        overlaps = overlaps[:max_overlapping_items]

        if overlaps:
            overlap_strs = []
            for frac, it in overlaps:
                tid   = it.get("id", "?")
                ttext = (it.get("text") or "").replace('"', "'")
                if len(ttext) > 40:
                    ttext = ttext[:37] + "..."
                overlap_strs.append(f'#{tid} "{ttext}"')
            overlap_info = " | overlaps OCR: " + ", ".join(overlap_strs)
        else:
            overlap_info = " | no OCR overlap"

        lines.append(
            f"  Region {i}: [{x1}, {y1}, {x2}, {y2}]{overlap_info}"
        )
    return "\n".join(lines)


def _format_ocr_compact(ocr_result: dict) -> str:
    items         = ocr_result.get("ocr_items", []) or []
    reading_order = (ocr_result.get("reading_order_text") or "").strip()
    if not items and not reading_order:
        return "No OCR text detected."
    parts = []
    if items:
        triplets = []
        for it in items:
            text = it.get("text", "")
            bbox = it.get("bbox", [])
            conf = round(float(it.get("confidence", 0.0)), 3)
            triplets.append(f'("{text}", {bbox}, {conf})')
        parts.append(
            "Detected words as (text, [bbox], confidence) triplets:\n\n"
            + ", ".join(triplets)
        )
    if reading_order:
        parts.append("Reading-order text:\n\n" + reading_order)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Annotated heatmap renderer (matches annotate_heatmaps.py)
# --------------------------------------------------------------------------- #
_BBOX_COLOR  = "#ff2e2e"
_HEATMAP_DPI = 150


def _render_annotated_heatmap(
    image_pil: Image.Image,
    prob: np.ndarray,
    dtd_regions: list,
    out_path: Path,
) -> Image.Image:
    """Build heatmap (jet overlay) and draw numbered red bboxes, matching
    annotate_heatmaps.py byte-for-byte."""
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
    base = (0.55 * img_arr + 0.45 * heat).clip(0, 255).astype(np.uint8)
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
    return Image.open(out_path).convert("RGB")


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
def _extract_clean_report(text: str) -> str:
    """Extract the final report. Handles <think>...</think><report>...</report>
    wrappers, bare anchored reports, and stripped reports."""
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text,
                  flags=re.DOTALL | re.IGNORECASE).strip()
    m = re.search(r"<report>(.*?)</report>", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    anchors = list(_REPORT_ANCHOR_RE.finditer(text))
    if anchors:
        text = text[anchors[-1].start():]
    end_idx = text.find(_END_MARKER)
    if end_idx >= 0:
        text = text[:end_idx + len(_END_MARKER)]
    return text.strip()


def _extract_think_block(text: str) -> Optional[str]:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "STAGE 1" in text.upper() or "STAGE 2" in text.upper():
        return text.strip()
    return None


_STUB_REPORT = """# FORGERY ANALYSIS REPORT

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 0

---

## DETAILED ANOMALY ANALYSIS

(no anomalies detected)

---

## SUMMARY
Model failed to produce a schema-compliant report.

**END OF REPORT**
"""


def _has_report_anchor(text: str) -> bool:
    return bool(_REPORT_ANCHOR_RE.search(text))


# --------------------------------------------------------------------------- #
# Stage extraction (used as fallback if stage-2 wants only KEEP regions)
# --------------------------------------------------------------------------- #
_STAGE1_ANCHOR_RE = re.compile(
    r"---\s*STAGE\s*1\s*[:.\-]\s*Knowledge\s*Preparation\s*---", re.IGNORECASE)
_STAGE2_ANCHOR_RE = re.compile(r"---\s*STAGE\s*2\s*[:.\-].*?---", re.IGNORECASE)
_STAGE3_ANCHOR_RE = re.compile(r"---\s*STAGE\s*3\s*[:.\-].*?---", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# DTD + OCR runners
# --------------------------------------------------------------------------- #
def _lazy_load_dtd(config_path: str, checkpoint_path: str, device):
    sys.path.insert(0, str(_DTD_SCRIPT_DIR))
    import run_doc_forensics_inference as _dtd  # noqa: E402
    _dtd._setup_paths_and_registry()
    dtd_model, dtd_model_name, dtd_needs_dct = _dtd.build_model_and_load(
        config_path, checkpoint_path, device,
    )
    return _dtd, dtd_model, dtd_model_name, dtd_needs_dct


def _run_dtd_for_image(image_path: Path, _dtd, dtd_model, dtd_model_name,
                        dtd_needs_dct, device, jpeg_quality: int = 95):
    prob, image_pil = _dtd.infer_one_image(
        image_path, dtd_model, dtd_model_name, dtd_needs_dct, device,
        jpeg_quality=jpeg_quality,
    )
    return prob, image_pil


def extract_ocr(
    image_path,
    *,
    gpu: bool = True,
    candidate_langs: list[str] | str = ("en", "ch", "th", "ms", "id", "ar"),
    mag_ratio: float = 1.0,
) -> dict:
    if isinstance(candidate_langs, str):
        candidate_langs = [s.strip() for s in candidate_langs.split(",")
                           if s.strip()]
    result = run_paddle_ocr_with_lang_detect(
        image_path, candidate_langs=candidate_langs, gpu=gpu,
        mag_ratio=mag_ratio, verbose=False,
    )
    result["selected_language"] = result["lang"]
    return result


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _find_prerendered_prompt(prompts_dir: Path, stem: str) -> Optional[Path]:
    """Look up {prompts_dir}/{stem}.prompt.txt (flat) or
    {prompts_dir}/{partXXX}/{stem}.prompt.txt (nested)."""
    flat = prompts_dir / f"{stem}.prompt.txt"
    if flat.exists():
        return flat
    for child in prompts_dir.iterdir():
        if child.is_dir() and child.name.lower().startswith("part"):
            p = child / f"{stem}.prompt.txt"
            if p.exists():
                return p
    return None


def _find_heatmap(heatmap_dir: Path, stem: str) -> Optional[Path]:
    """Look up annotated heatmap PNG. Tries the same naming patterns as
    train_student_sft.py / annotate_heatmaps.py:
        {dir}/{stem}.heatmap_annotated.png
        {dir}/{stem}.heatmap.png
        {dir}/{stem}_dtd.png  ...
    plus nested partXXX/ layout."""
    cands_flat = [
        f"{stem}.heatmap_annotated.png",
        f"{stem}.heatmap.png",
        f"{stem}_dtd.png",
        f"{stem}.dtd.png",
        f"{stem}_heatmap.png",
    ]
    for name in cands_flat:
        p = heatmap_dir / name
        if p.exists():
            return p
    for child in heatmap_dir.iterdir():
        if child.is_dir() and child.name.lower().startswith("part"):
            for name in cands_flat:
                p = child / name
                if p.exists():
                    return p
    return None


# --------------------------------------------------------------------------- #
# Smart-resize math (matches Qwen3-VL processor)
# --------------------------------------------------------------------------- #
def _smart_resize_dims(w: int, h: int, max_pixels: int) -> tuple[int, int]:
    if max_pixels <= 0:
        return w, h
    if w * h <= max_pixels:
        return (round(w / 28) * 28, round(h / 28) * 28)
    ratio = math.sqrt(max_pixels / (w * h))
    new_w = max(28, round(w * ratio / 28) * 28)
    new_h = max(28, round(h * ratio / 28) * 28)
    while new_w * new_h > max_pixels and new_w > 28:
        new_w -= 28
    while new_w * new_h > max_pixels and new_h > 28:
        new_h -= 28
    return new_w, new_h


# --------------------------------------------------------------------------- #
# Visualisations
# --------------------------------------------------------------------------- #
def _draw_boxes(
    image: Image.Image, report: ForgeryReport, out_path: Path, title: str = "",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.parse_math"] = False
    import matplotlib.patches as mpatches

    w, h = image.size
    fig_w = min(16, max(6, w / 100))
    fig_h = max(4, fig_w * h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    for a in report.anomalies:
        if not a.grounding or len(a.grounding) != 4:
            continue
        x1, y1, x2, y2 = a.grounding
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        rect = mpatches.Rectangle(
            (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
            linewidth=2, edgecolor="#ff2e2e", facecolor="none",
        )
        ax.add_patch(rect)
        label_bits = [f"#{a.index}"]
        if a.type:
            label_bits.append(a.type[:28])
        ax.text(
            x1, max(0, y1 - 6), "  ".join(label_bits),
            color="white", fontsize=9,
            bbox=dict(facecolor="#ff2e2e", edgecolor="none", alpha=0.9, pad=2),
        )
    try:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    except Exception as exc:
        print(f"  [warn] tight_layout failed ({exc!r}); saving without it")
        try:
            fig.savefig(out_path, dpi=150)
        except Exception as exc2:
            print(f"  [warn] savefig also failed: {exc2!r}")
    plt.close(fig)


def _find_gt_mask(masks_dir: Optional[Path], stem: str) -> Optional[Path]:
    if masks_dir is None or not masks_dir.is_dir():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        for cand in (masks_dir / f"{stem}{ext}",
                     masks_dir / f"{stem}_mask{ext}"):
            if cand.exists():
                return cand
    return None


def _save_viz_input_artifacts(
    viz_dir: Path,
    stem: str,
    image_pil: Image.Image,
    heatmap_pil: Optional[Image.Image],
    prompt: str,
    max_image_pixels: int,
    subdir: Optional[str] = None,
) -> None:
    """Save what the model will see, AFTER smart_resize. Mirrors
    train_student_sft.py CotJsonlDataset._save_viz_artifacts."""
    base = viz_dir / subdir if subdir else viz_dir
    sample_dir = base / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    orig_w, orig_h = image_pil.size
    proc_w, proc_h = _smart_resize_dims(orig_w, orig_h, max_image_pixels)

    img = image_pil
    if (proc_w, proc_h) != (orig_w, orig_h):
        img = image_pil.resize((proc_w, proc_h), Image.BILINEAR)
    img.save(sample_dir / f"{stem}_input_image.png")

    if heatmap_pil is not None:
        hm = heatmap_pil
        if hm.size != (proc_w, proc_h):
            hm = hm.resize((proc_w, proc_h), Image.BILINEAR)
        hm.save(sample_dir / f"{stem}_input_heatmap.png")

    (sample_dir / f"{stem}_input_prompt.txt").write_text(
        prompt, encoding="utf-8")
    (sample_dir / f"{stem}_meta.json").write_text(
        json.dumps({
            "original_size":   [orig_w, orig_h],
            "processed_size":  [proc_w, proc_h],
            "max_image_pixels": max_image_pixels,
            "has_heatmap":     heatmap_pil is not None,
            "subdir":          subdir,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Model loaders
# --------------------------------------------------------------------------- #
def _load_base_model(args, max_image_pixels: int):
    """Load base Qwen3-VL and its processor (no adapter yet).
    Returns (model, processor)."""
    import torch
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig

    print(f"[load] base {args.base_model_id} "
          f"(max_pixels={max_image_pixels})", flush=True)
    proc_kwargs = {"trust_remote_code": True}
    if max_image_pixels > 0:
        proc_kwargs["max_pixels"] = max_image_pixels
    processor = AutoProcessor.from_pretrained(args.base_model_id, **proc_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ModelCls = getattr(transformers, args.model_class, None)
    if ModelCls is None:
        from transformers import AutoModelForVision2Seq
        ModelCls = AutoModelForVision2Seq

    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        model = ModelCls.from_pretrained(
            args.base_model_id, quantization_config=bnb,
            dtype=torch.bfloat16, device_map="auto",
            attn_implementation=args.attn_impl, trust_remote_code=True,
        )
    else:
        model = ModelCls.from_pretrained(
            args.base_model_id, dtype=torch.bfloat16,
            device_map="auto", attn_implementation=args.attn_impl,
            trust_remote_code=True,
        )
    return model, processor


def _attach_adapter(base_model, adapter_path: Optional[str]):
    """Wrap base_model in a PeftModel with the given LoRA adapter. Returns
    the wrapped model (eval mode). If adapter_path is None, returns base
    unwrapped (zero-shot)."""
    if adapter_path is None:
        base_model.eval()
        return base_model
    from peft import PeftModel
    print(f"[load] adapter {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def _set_processor_max_pixels(processor, max_pixels: int) -> None:
    """Mutate Qwen3-VL processor's image-processor max_pixels in place."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["max_pixels"] = max_pixels


# --------------------------------------------------------------------------- #
# Generation helper
# --------------------------------------------------------------------------- #
def _build_gen_kwargs(args, processor) -> dict:
    kwargs = dict(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        pad_token_id=(processor.tokenizer.pad_token_id
                      or processor.tokenizer.eos_token_id),
    )
    if not args.greedy:
        kwargs["temperature"] = max(args.temperature, 1e-5)
        kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            kwargs["top_k"] = args.top_k
    return kwargs


def _generate(model, processor, messages: list[dict],
              gen_kwargs: dict, stream: bool) -> str:
    """apply_chat_template -> generate -> decode. Returns raw decoded text."""
    import torch
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}

    kwargs = dict(gen_kwargs)
    if stream:
        kwargs["streamer"] = TextStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **kwargs)

    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]


# --------------------------------------------------------------------------- #
# Prompt building (live mode)
# --------------------------------------------------------------------------- #
def _append_image_metadata(prompt: str, orig_w: int, orig_h: int,
                            append: bool) -> str:
    """Append IMAGE METADATA suffix if not already present in the prompt."""
    if not append:
        return prompt
    if "IMAGE METADATA" in prompt:
        return prompt
    return prompt + (
        f"\n\nIMAGE METADATA:\n"
        f"- Width: {orig_w} pixels\n"
        f"- Height: {orig_h} pixels\n"
        f"- Coordinate system: top-left origin, x grows right, y grows down.\n"
        f"- All [GROUNDING] values MUST be absolute integer pixel "
        f"coordinates in this {orig_w}x{orig_h} image."
    )


def _build_live_prompt_all(template: str, ocr_result: dict,
                            dtd_regions: list) -> str:
    """--regime_all live prompt: substitute {{OCR_JSON}} and {{DTD_HINTS}}."""
    return (
        template
        .replace("{{OCR_JSON}}",
                  _format_ocr_compact(ocr_result))
        .replace("{{DTD_HINTS}}",
                  _format_dtd_hints_with_overlap(
                      dtd_regions, ocr_result.get("ocr_items", []),
                  ))
    )


def _build_live_prompt_stage1(template: str, ocr_result: dict,
                                dtd_regions: list) -> str:
    """Two-stage stage 1 live prompt: only {{DTD_HINTS}} is substituted.
    OCR text appears inside DTD-region overlap info — stage 1 template has
    no {{OCR_JSON}} slot."""
    return template.replace(
        "{{DTD_HINTS}}",
        _format_dtd_hints_with_overlap(
            dtd_regions, ocr_result.get("ocr_items", []),
        ),
    )


def _build_live_prompt_stage2(template: str, ocr_result: dict,
                                stage1_full_output: str) -> str:
    """Two-stage stage 2 live prompt: substitute {{OCR_JSON}} (compact) and
    {{FILTERED_DTD}} (the FULL stage-1 output verbatim — both STAGE 1 and
    STAGE 2 reasoning, no extra filtering)."""
    return (
        template
        .replace("{{OCR_JSON}}",     _format_ocr_compact(ocr_result))
        .replace("{{FILTERED_DTD}}", stage1_full_output)
    )


def _build_folder_prompt_stage2(prompt_template_text: str,
                                  stage1_full_output: str) -> str:
    """Pre-rendered stage-2 prompt: substitute ONLY {{FILTERED_DTD}}. OCR
    and image metadata are already baked in at prerender time."""
    return prompt_template_text.replace("{{FILTERED_DTD}}", stage1_full_output)


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #
def process_one_image_all(
    image_path: Path,
    *,
    args,
    model, processor,
    dtd_bundle,
    prompt_mode: str,  # "template" or "folder"
    prompt_template_text: Optional[str],
    prompt_folder: Optional[Path],
    heatmap_dir: Optional[Path],
    out_dir: Path,
    viz_inputs_dir: Optional[Path],
    viz_inputs_remaining: list[int],
) -> None:
    """--regime_all: one-pass inference with --adapter_path (or zero-shot).

    Prompt source:
      * 'template': run OCR + DTD live, build heatmap with numbered bboxes,
        substitute {{OCR_JSON}} and {{DTD_HINTS}} into the template.
      * 'folder':   load {stem}.prompt.txt verbatim, load annotated heatmap
        from --heatmap_dir, skip OCR/DTD entirely (unless --viz_outputs is
        on, in which case OCR runs just for ocr_viz.png).
    """
    import torch

    t_start = time.time()
    stem = image_path.stem
    doc_dir = out_dir / stem

    if getattr(args, "skip_existing", False) and (doc_dir / "report.md").exists():
        print(f"\n{'='*60}\n{image_path.name}\n{'='*60}")
        print(f"  [skip] report.md already exists")
        return

    doc_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n{image_path.name}  [regime_all / "
          f"prompt={prompt_mode}]\n{'='*60}")

    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    # ====================================================================== #
    # Build prompt + heatmap according to prompt_mode
    # ====================================================================== #
    ocr_result: Optional[dict] = None  # populated unconditionally if viz_outputs
    heatmap_pil: Optional[Image.Image] = None

    if prompt_mode == "template":
        # ---- DTD (live) ----
        if dtd_bundle is None:
            raise SystemExit(
                "--prompt_template requires DTD (provide --dtd_config + "
                "--dtd_checkpoint)."
            )
        _dtd_mod, dtd_model, dtd_model_name, dtd_needs_dct, device = dtd_bundle
        print("  [dtd] running ...", flush=True)
        prob, _ = _run_dtd_for_image(
            image_path, _dtd_mod, dtd_model, dtd_model_name,
            dtd_needs_dct, device, jpeg_quality=args.jpeg_quality,
        )
        dtd_regions = _dtd_regions_from_prob(
            prob, threshold=args.dtd_threshold, min_area=200,
        )
        print(f"  [dtd] {len(dtd_regions)} region(s) at threshold "
              f"{args.dtd_threshold}")
        if args.save_dtd_probs:
            np.save(doc_dir / "dtd.prob.npy", prob.astype(np.float32))

        # ---- OCR (live, with language detect) ----
        print("  [ocr] running ...", flush=True)
        ocr_result = extract_ocr(
            image_path,
            gpu=torch.cuda.is_available(),
            candidate_langs=args.langs,
        )

        # ---- Heatmap (live, annotated with numbered red bboxes) ----
        heatmap_pil = _render_annotated_heatmap(
            image_pil=image_pil, prob=prob, dtd_regions=dtd_regions,
            out_path=doc_dir / "dtd_overlay.png",
        )
        if heatmap_pil.size != image_pil.size:
            heatmap_pil = heatmap_pil.resize(image_pil.size, Image.BILINEAR)

        # ---- Prompt (live build) ----
        prompt = _build_live_prompt_all(
            prompt_template_text, ocr_result, dtd_regions,
        )
        prompt = _append_image_metadata(
            prompt, orig_w, orig_h, args.append_image_metadata,
        )

    elif prompt_mode == "folder":
        # ---- Prompt: pre-rendered ----
        pp_path = _find_prerendered_prompt(prompt_folder, stem)
        if pp_path is None:
            print(f"  [skip] no pre-rendered prompt for {stem} in "
                  f"{prompt_folder}")
            return
        prompt = pp_path.read_text(encoding="utf-8")
        print(f"  [prompt] loaded prerendered: {pp_path.relative_to(prompt_folder)}")

        # ---- Heatmap: from disk ----
        if heatmap_dir is not None:
            hm_path = _find_heatmap(heatmap_dir, stem)
            if hm_path is None:
                print(f"  [skip] no heatmap for {stem} in {heatmap_dir}")
                return
            heatmap_pil = Image.open(hm_path).convert("RGB")
            if heatmap_pil.size != image_pil.size:
                heatmap_pil = heatmap_pil.resize(image_pil.size, Image.BILINEAR)
            # Save a copy for inspection
            heatmap_pil.save(doc_dir / "dtd_overlay.png")
            print(f"  [heatmap] loaded from {hm_path.name}")
        else:
            # Allowed: model trained without heatmap input. No-op.
            pass

        # ---- OCR for viz only (no DTD needed) ----
        if args.viz_outputs:
            print("  [ocr] running (for viz only) ...", flush=True)
            try:
                ocr_result = extract_ocr(
                    image_path,
                    gpu=torch.cuda.is_available(),
                    candidate_langs=args.langs,
                )
            except Exception as exc:
                print(f"  [warn] ocr for viz failed: {exc!r}")
                ocr_result = None
    else:
        raise SystemExit(f"unknown prompt_mode: {prompt_mode}")

    # ====================================================================== #
    # OCR viz (if available)
    # ====================================================================== #
    if args.viz_outputs and ocr_result is not None:
        try:
            draw_ocr_boxes(
                image_path, ocr_result["ocr_items"],
                doc_dir / "ocr_viz.png",
                title=(f"OCR ({ocr_result.get('lang', '?')}) - "
                       f"{ocr_result.get('n_items', len(ocr_result.get('ocr_items', [])))} items"),
            )
        except Exception as exc:
            print(f"  [warn] ocr_viz failed: {exc!r}")

    # ====================================================================== #
    # Save final prompt + optional input-viz dump
    # ====================================================================== #
    (doc_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    use_heatmap = heatmap_pil is not None
    if viz_inputs_dir is not None and viz_inputs_remaining[0] > 0:
        try:
            _save_viz_input_artifacts(
                viz_dir=viz_inputs_dir, stem=stem,
                image_pil=image_pil,
                heatmap_pil=heatmap_pil if use_heatmap else None,
                prompt=prompt,
                max_image_pixels=args.max_image_pixels,
            )
            viz_inputs_remaining[0] -= 1
        except Exception as exc:
            print(f"  [warn] viz_inputs dump failed: {exc!r}")

    # ====================================================================== #
    # Build messages and generate
    # ====================================================================== #
    user_content = [{"type": "image", "image": image_pil}]
    if use_heatmap:
        user_content.append({"type": "image", "image": heatmap_pil})
    user_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": user_content}]

    n_images = sum(1 for c in user_content if c["type"] == "image")
    print(f"  [gen] images={n_images}  prompt_chars={len(prompt)}  "
          f"max_new={args.max_new_tokens}", flush=True)
    t0 = time.time()
    gen_kwargs = _build_gen_kwargs(args, processor)
    raw_text = _generate(model, processor, messages, gen_kwargs,
                          stream=args.stream)
    elapsed = time.time() - t0
    print(f"\n  [gen] {elapsed:.1f}s  {len(raw_text)} chars", flush=True)

    # ====================================================================== #
    # Save outputs + parse report
    # ====================================================================== #
    (doc_dir / "report.raw.txt").write_text(raw_text, encoding="utf-8")

    think = _extract_think_block(raw_text)
    if think:
        (doc_dir / "think.txt").write_text(think, encoding="utf-8")

    answer = _extract_clean_report(raw_text)
    if not _has_report_anchor(answer):
        print("  [WARNING] Missing '# FORGERY ANALYSIS REPORT'. Using stub.")
        answer = _STUB_REPORT

    report = parse_report(answer)
    (doc_dir / "report.md").write_text(
        answer + ("\n" if not answer.endswith("\n") else ""), encoding="utf-8",
    )
    (doc_dir / "report.json").write_text(
        json.dumps({"image_name": image_path.name, "report": answer},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.viz_outputs:
        _emit_output_viz(
            args, image_path, image_pil, report, doc_dir, stem,
        )

    total = time.time() - t_start
    print(
        f"\n  [done] verdict={report.conclusion}  score={report.risk_score}  "
        f"anomalies={len(report.anomalies)}  gen={elapsed:.1f}s  "
        f"total={total:.1f}s"
    )
    for a in report.anomalies:
        print(f"    #{a.index:03d}  {(a.type or '?'):<30}  "
              f"grounding={a.grounding}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_one_image_two_stage(
    image_path: Path,
    *,
    args,
    stage1_model, stage1_processor,
    stage2_model, stage2_processor,
    switch_to_stage2,  # callable() -> None, invoked just before stage-2 gen
    dtd_bundle,
    prompt_mode_stage1: str, prompt_mode_stage2: str,
    stage1_template_text: Optional[str],
    stage2_template_text: Optional[str],
    stage1_prompt_folder: Optional[Path],
    stage2_prompt_folder: Optional[Path],
    heatmap_dir: Optional[Path],
    out_dir: Path,
    viz_inputs_dir: Optional[Path],
    viz_inputs_remaining: list[int],
) -> None:
    """--regime_two_stage: stage 1 produces full STAGE 1+2 reasoning, stage 2
    plugs that text into {{FILTERED_DTD}} of the stage-2 prompt and produces
    the final report."""
    import torch

    t_start = time.time()
    stem = image_path.stem
    doc_dir = out_dir / stem

    if getattr(args, "skip_existing", False) and (doc_dir / "report.md").exists():
        print(f"\n{'='*60}\n{image_path.name}\n{'='*60}")
        print(f"  [skip] report.md already exists")
        return

    doc_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n{image_path.name}  [regime_two_stage / "
          f"stage1={prompt_mode_stage1} stage2={prompt_mode_stage2}]\n{'='*60}")

    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    # ====================================================================== #
    # STAGE 1: image + heatmap + stage1 prompt
    # ====================================================================== #
    print("\n  -- STAGE 1: DTD region reasoning --", flush=True)

    ocr_result: Optional[dict] = None
    heatmap_pil: Optional[Image.Image] = None

    if prompt_mode_stage1 == "template":
        if dtd_bundle is None:
            raise SystemExit(
                "Stage 1 with --prompt_template requires DTD."
            )
        _dtd_mod, dtd_model, dtd_model_name, dtd_needs_dct, device = dtd_bundle
        print("  [dtd] running ...", flush=True)
        prob, _ = _run_dtd_for_image(
            image_path, _dtd_mod, dtd_model, dtd_model_name,
            dtd_needs_dct, device, jpeg_quality=args.jpeg_quality,
        )
        dtd_regions = _dtd_regions_from_prob(
            prob, threshold=args.dtd_threshold, min_area=200,
        )
        print(f"  [dtd] {len(dtd_regions)} region(s) at threshold "
              f"{args.dtd_threshold}")
        if args.save_dtd_probs:
            np.save(doc_dir / "dtd.prob.npy", prob.astype(np.float32))

        print("  [ocr] running ...", flush=True)
        ocr_result = extract_ocr(
            image_path,
            gpu=torch.cuda.is_available(),
            candidate_langs=args.langs,
        )

        heatmap_pil = _render_annotated_heatmap(
            image_pil=image_pil, prob=prob, dtd_regions=dtd_regions,
            out_path=doc_dir / "dtd_overlay.png",
        )
        if heatmap_pil.size != image_pil.size:
            heatmap_pil = heatmap_pil.resize(image_pil.size, Image.BILINEAR)

        stage1_prompt = _build_live_prompt_stage1(
            stage1_template_text, ocr_result, dtd_regions,
        )
        stage1_prompt = _append_image_metadata(
            stage1_prompt, orig_w, orig_h, args.append_image_metadata,
        )

    elif prompt_mode_stage1 == "folder":
        pp = _find_prerendered_prompt(stage1_prompt_folder, stem)
        if pp is None:
            print(f"  [skip] no stage-1 pre-rendered prompt for {stem}")
            return
        stage1_prompt = pp.read_text(encoding="utf-8")
        print(f"  [stage1 prompt] {pp.relative_to(stage1_prompt_folder)}")

        if heatmap_dir is None:
            raise SystemExit(
                "Stage 1 with --prompt_folder_stage1 requires --heatmap_dir."
            )
        hm_path = _find_heatmap(heatmap_dir, stem)
        if hm_path is None:
            print(f"  [skip] no heatmap for {stem} in {heatmap_dir}")
            return
        heatmap_pil = Image.open(hm_path).convert("RGB")
        if heatmap_pil.size != image_pil.size:
            heatmap_pil = heatmap_pil.resize(image_pil.size, Image.BILINEAR)
        heatmap_pil.save(doc_dir / "dtd_overlay.png")

        if args.viz_outputs:
            try:
                ocr_result = extract_ocr(
                    image_path,
                    gpu=torch.cuda.is_available(),
                    candidate_langs=args.langs,
                )
            except Exception as exc:
                print(f"  [warn] ocr for viz failed: {exc!r}")
                ocr_result = None
    else:
        raise SystemExit(f"unknown stage-1 prompt_mode: {prompt_mode_stage1}")

    (doc_dir / "stage1_prompt.txt").write_text(stage1_prompt, encoding="utf-8")

    # OCR viz once (if we have ocr_result)
    if args.viz_outputs and ocr_result is not None:
        try:
            draw_ocr_boxes(
                image_path, ocr_result["ocr_items"],
                doc_dir / "ocr_viz.png",
                title=(f"OCR ({ocr_result.get('lang', '?')}) - "
                       f"{ocr_result.get('n_items', len(ocr_result.get('ocr_items', [])))} items"),
            )
        except Exception as exc:
            print(f"  [warn] ocr_viz failed: {exc!r}")

    # Input-viz dump for stage 1
    if viz_inputs_dir is not None and viz_inputs_remaining[0] > 0:
        try:
            _save_viz_input_artifacts(
                viz_dir=viz_inputs_dir, stem=stem,
                image_pil=image_pil, heatmap_pil=heatmap_pil,
                prompt=stage1_prompt,
                max_image_pixels=args.max_image_pixels_stage1,
                subdir="stage1",
            )
        except Exception as exc:
            print(f"  [warn] stage1 viz_inputs failed: {exc!r}")

    # Generate stage 1
    stage1_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "image", "image": heatmap_pil},
            {"type": "text",  "text":  stage1_prompt},
        ],
    }]
    print(f"  [gen stage1] images=2  prompt_chars={len(stage1_prompt)}",
          flush=True)
    gen_kwargs = _build_gen_kwargs(args, stage1_processor)
    t1 = time.time()
    stage1_raw = _generate(stage1_model, stage1_processor, stage1_messages,
                            gen_kwargs, stream=args.stream)
    stage1_elapsed = time.time() - t1
    (doc_dir / "stage1_raw.txt").write_text(stage1_raw, encoding="utf-8")

    # Keep the entire stage-1 output for {{FILTERED_DTD}} (per user spec:
    # "Этот результат нужно взять весь ничего не удаляя").
    # We do strip any wrapping <think>...</think> tags so the substituted
    # text reads as plain reasoning, but the body itself is unmodified.
    stage1_think_only = _extract_think_block(stage1_raw)
    filtered_dtd_text = stage1_think_only if stage1_think_only else stage1_raw

    (doc_dir / "stage1_reasoning.txt").write_text(
        filtered_dtd_text, encoding="utf-8")
    print(f"\n  [stage1] {stage1_elapsed:.1f}s  {len(stage1_raw)} chars  "
          f"-> FILTERED_DTD={len(filtered_dtd_text)} chars")

    # ====================================================================== #
    # STAGE 2: image only + stage2 prompt (with {{FILTERED_DTD}} filled in)
    # ====================================================================== #
    print("\n  -- STAGE 2: Final report --", flush=True)

    if prompt_mode_stage2 == "template":
        if stage2_template_text is None:
            raise SystemExit("--prompt_template_stage2 is required")
        # Run OCR live if we don't already have it (stage-1 folder mode)
        if ocr_result is None:
            print("  [ocr] running for stage 2 ...", flush=True)
            ocr_result = extract_ocr(
                image_path,
                gpu=torch.cuda.is_available(),
                candidate_langs=args.langs,
            )
        stage2_prompt = _build_live_prompt_stage2(
            stage2_template_text, ocr_result, filtered_dtd_text,
        )
        stage2_prompt = _append_image_metadata(
            stage2_prompt, orig_w, orig_h, args.append_image_metadata,
        )

    elif prompt_mode_stage2 == "folder":
        pp2 = _find_prerendered_prompt(stage2_prompt_folder, stem)
        if pp2 is None:
            print(f"  [skip] no stage-2 pre-rendered prompt for {stem}")
            return
        # Pre-rendered stage-2 prompt has {{FILTERED_DTD}} placeholder still
        # in it (rendered at prerender time with no live stage-1 output).
        # Per user spec: re-substitute with the just-produced stage-1 output.
        stage2_prompt_text_raw = pp2.read_text(encoding="utf-8")
        stage2_prompt = _build_folder_prompt_stage2(
            stage2_prompt_text_raw, filtered_dtd_text,
        )
        print(f"  [stage2 prompt] {pp2.relative_to(stage2_prompt_folder)}")
    else:
        raise SystemExit(f"unknown stage-2 prompt_mode: {prompt_mode_stage2}")

    (doc_dir / "stage2_prompt.txt").write_text(stage2_prompt, encoding="utf-8")

    # Input-viz dump for stage 2 (NO heatmap input by training spec)
    if viz_inputs_dir is not None and viz_inputs_remaining[0] > 0:
        try:
            _save_viz_input_artifacts(
                viz_dir=viz_inputs_dir, stem=stem,
                image_pil=image_pil, heatmap_pil=None,
                prompt=stage2_prompt,
                max_image_pixels=args.max_image_pixels_stage2,
                subdir="stage2",
            )
            viz_inputs_remaining[0] -= 1
        except Exception as exc:
            print(f"  [warn] stage2 viz_inputs failed: {exc!r}")

    # Retarget shared processor's max_pixels if stage-2 differs
    if (stage2_processor is stage1_processor
            and args.max_image_pixels_stage2 != args.max_image_pixels_stage1):
        try:
            _set_processor_max_pixels(stage1_processor,
                                       args.max_image_pixels_stage2)
            print(f"  [proc] retargeted max_pixels="
                  f"{args.max_image_pixels_stage2} for stage 2")
        except Exception as exc:
            print(f"  [warn] could not retarget max_pixels: {exc!r}")

    # Switch active LoRA adapter to stage 2 (no-op if separate models)
    try:
        switch_to_stage2()
    except Exception as exc:
        print(f"  [warn] switch_to_stage2 failed: {exc!r}")

    stage2_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text",  "text":  stage2_prompt},
        ],
    }]
    print(f"  [gen stage2] images=1  prompt_chars={len(stage2_prompt)}",
          flush=True)
    gen_kwargs = _build_gen_kwargs(args, stage2_processor)
    t2 = time.time()
    stage2_raw = _generate(stage2_model, stage2_processor, stage2_messages,
                            gen_kwargs, stream=args.stream)
    stage2_elapsed = time.time() - t2
    (doc_dir / "stage2_raw.txt").write_text(stage2_raw, encoding="utf-8")
    (doc_dir / "report.raw.txt").write_text(stage2_raw, encoding="utf-8")

    # Restore max_pixels for next iteration if we mutated it
    if (stage2_processor is stage1_processor
            and args.max_image_pixels_stage2 != args.max_image_pixels_stage1):
        try:
            _set_processor_max_pixels(stage1_processor,
                                       args.max_image_pixels_stage1)
        except Exception:
            pass

    # ---- Parse <report>...</report> ----
    answer = _extract_clean_report(stage2_raw)
    if not _has_report_anchor(answer):
        print("  [WARNING] Missing '# FORGERY ANALYSIS REPORT'. Using stub.")
        answer = _STUB_REPORT

    stage2_think = _extract_think_block(stage2_raw)
    if stage2_think:
        (doc_dir / "stage2_reasoning.txt").write_text(
            stage2_think, encoding="utf-8")

    report = parse_report(answer)
    (doc_dir / "report.md").write_text(
        answer + ("\n" if not answer.endswith("\n") else ""), encoding="utf-8",
    )
    (doc_dir / "report.json").write_text(
        json.dumps({"image_name": image_path.name, "report": answer},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.viz_outputs:
        _emit_output_viz(
            args, image_path, image_pil, report, doc_dir, stem,
        )

    total = time.time() - t_start
    print(
        f"\n  [done] verdict={report.conclusion}  score={report.risk_score}  "
        f"anomalies={len(report.anomalies)}  "
        f"stage1={stage1_elapsed:.1f}s  stage2={stage2_elapsed:.1f}s  "
        f"total={total:.1f}s"
    )
    for a in report.anomalies:
        print(f"    #{a.index:03d}  {(a.type or '?'):<30}  "
              f"grounding={a.grounding}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _emit_output_viz(args, image_path: Path, image_pil: Image.Image,
                      report: ForgeryReport, doc_dir: Path, stem: str) -> None:
    """Shared output-visualisation logic: report_viz.png, pred_viz.png, gt_viz.png."""
    viz_title = (
        f"{image_path.name}   |   {report.conclusion}   |   "
        f"score={report.risk_score}   |   anomalies={len(report.anomalies)}"
    )
    _draw_boxes(image_pil, report, doc_dir / "report_viz.png", title=viz_title)

    if _HAS_VIS_REPORT:
        gt_mask_path = (
            _find_gt_mask(
                Path(args.gt_masks_dir).expanduser().resolve(), stem,
            ) if args.gt_masks_dir else None
        )
        try:
            if gt_mask_path:
                visualize_report_with_mask(
                    image_path, doc_dir / "report.md",
                    doc_dir / "pred_viz.png",
                    gt_mask_path=gt_mask_path,
                    title=f"Pipeline (mask bboxes): {image_path.name}",
                )
            else:
                visualize_report(
                    image_path, doc_dir / "report.md",
                    doc_dir / "pred_viz.png",
                    title=f"Pipeline: {image_path.name}",
                )
        except Exception as exc:
            print(f"  [warn] pred_viz failed: {exc!r}")

    if _HAS_VIS_REPORT and args.gt_reports:
        gt_base = Path(args.gt_reports).expanduser().resolve()
        gt_path = (gt_base if gt_base.is_file()
                   else gt_base / f"{stem}_report.md")
        if gt_path.exists():
            try:
                visualize_report(
                    image_path, gt_path,
                    doc_dir / "gt_viz.png",
                    title=f"Ground Truth: {image_path.name}",
                )
            except Exception as exc:
                print(f"  [warn] gt_viz failed: {exc!r}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Regime (mutex) ---
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regime_all", action="store_true",
                   help="One-pass inference: image (+ optional heatmap) + "
                        "single prompt -> final report. Needs --adapter_path "
                        "(or run zero-shot if omitted).")
    g.add_argument("--regime_two_stage", action="store_true",
                   help="Two-pass inference: stage 1 produces STAGE 1+2 "
                        "reasoning, stage 2 consumes it as {{FILTERED_DTD}} "
                        "and produces the final report. Needs both "
                        "--adapter_stage1_path and --adapter_stage2_path.")

    # --- Base model ---
    ap.add_argument("--base_model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--model_class", default="Qwen3VLForConditionalGeneration")
    ap.add_argument("--attn_impl", default="sdpa",
                    choices=("eager", "sdpa", "flash_attention_2"))
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Load base in 4-bit (matches QLoRA training).")

    # --- Adapters ---
    ap.add_argument("--adapter_path", default=None,
                    help="(--regime_all) Path to LoRA adapter dir "
                         "(adapter_config.json + adapter_model.safetensors). "
                         "Omit to run zero-shot on base.")
    ap.add_argument("--adapter_stage1_path", default=None,
                    help="(--regime_two_stage) Required. Stage-1 LoRA adapter.")
    ap.add_argument("--adapter_stage2_path", default=None,
                    help="(--regime_two_stage) Required. Stage-2 LoRA adapter.")

    # --- Image source ---
    ig = ap.add_mutually_exclusive_group(required=True)
    ig.add_argument("--image", help="Single image path.")
    ig.add_argument("--image_dir", help="Directory of images.")

    # --- Prompt source for --regime_all ---
    ap.add_argument("--prompt_template", default=None,
                    help="(--regime_all) Template with {{OCR_JSON}} and "
                         "{{DTD_HINTS}} placeholders. Triggers live OCR + "
                         "live DTD + live annotated heatmap.")
    ap.add_argument("--prompt_folder", default=None,
                    help="(--regime_all) Folder of pre-rendered "
                         "{stem}.prompt.txt files. Requires --heatmap_dir.")

    # --- Prompt source for --regime_two_stage ---
    ap.add_argument("--prompt_template_stage1", default=None,
                    help="(--regime_two_stage) Stage-1 template with "
                         "{{DTD_HINTS}}.")
    ap.add_argument("--prompt_template_stage2", default=None,
                    help="(--regime_two_stage) Stage-2 template with "
                         "{{OCR_JSON}} and {{FILTERED_DTD}}.")
    ap.add_argument("--prompt_folder_stage1", default=None,
                    help="(--regime_two_stage) Folder of pre-rendered "
                         "stage-1 prompts. Requires --heatmap_dir.")
    ap.add_argument("--prompt_folder_stage2", default=None,
                    help="(--regime_two_stage) Folder of pre-rendered "
                         "stage-2 prompts. The {{FILTERED_DTD}} placeholder "
                         "in these will be re-filled with live stage-1 output.")

    # --- Heatmap source (only for --prompt_folder modes) ---
    ap.add_argument("--heatmap_dir", default=None,
                    help="Folder of pre-annotated heatmap PNGs. Required "
                         "when --prompt_folder (regime_all) or "
                         "--prompt_folder_stage1 (regime_two_stage) is used.")

    # --- DTD (live, used in --prompt_template modes) ---
    ap.add_argument("--dtd_config", default=None,
                    help="DTD YAML config. Required if any --prompt_template* "
                         "is in use.")
    ap.add_argument("--dtd_checkpoint", default=None,
                    help="DTD checkpoint .pth. Required if any "
                         "--prompt_template* is in use.")
    ap.add_argument("--dtd_threshold", type=float, default=0.98,
                    help="DTD probability threshold for the region bboxes.")

    # --- OCR (live, used in --prompt_template modes; also for viz with folder mode) ---
    ap.add_argument("--langs", default="en,ch,th,ms,id,ar")

    # --- Image resolution (must match training) ---
    ap.add_argument("--max_image_pixels", type=int, default=540000,
                    help="(--regime_all) max_pixels for the single pass. "
                         "MUST match the value used during training of the "
                         "loaded adapter.")
    ap.add_argument("--max_image_pixels_stage1", type=int, default=1800000,
                    help="(--regime_two_stage) max_pixels for stage 1.")
    ap.add_argument("--max_image_pixels_stage2", type=int, default=600000,
                    help="(--regime_two_stage) max_pixels for stage 2.")

    # --- Image metadata suffix ---
    ap.add_argument("--append_image_metadata", action="store_true",
                    default=True,
                    help="Append 'IMAGE METADATA' suffix (orig W/H + coord "
                         "system) when not already in prompt. Default ON.")
    ap.add_argument("--no_append_image_metadata",
                    dest="append_image_metadata", action="store_false")

    # --- Generation ---
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--min_new_tokens", type=int, default=64)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--stream", action="store_true",
                    help="Stream tokens to stdout via TextStreamer.")

    # --- Output ---
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--save_dtd_probs", action="store_true")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip an image if {out_dir}/{stem}/report.md exists.")

    # --- Viz ---
    ap.add_argument("--viz_outputs", action="store_true", default=True,
                    help="Save output artifacts (ocr_viz, report_viz, "
                         "pred_viz, gt_viz). Default ON.")
    ap.add_argument("--no_viz_outputs", dest="viz_outputs", action="store_false")
    ap.add_argument("--viz_inputs", default=None,
                    help="If set, save per-sample INPUT artifacts (resized "
                         "image + resized heatmap + rendered prompt + meta) "
                         "to this dir. Mirrors train_student_sft.py.")
    ap.add_argument("--viz_inputs_max_samples", type=int, default=40,
                    help="How many --viz_inputs samples to dump (default 40).")

    # --- GT for viz ---
    ap.add_argument("--gt_reports", default=None,
                    help="Path to GT report file or directory for gt_viz.png.")
    ap.add_argument("--gt_masks_dir", default=None,
                    help="Dir of GT masks for mask-based pred_viz.")

    return ap.parse_args()


def gather_images(args) -> list[Path]:
    if args.image:
        return [Path(args.image).expanduser().resolve()]
    d = Path(args.image_dir).expanduser().resolve()
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)

    if args.skip_existing:
        out_dir = Path(args.out_dir).expanduser().resolve()
        before = len(paths)
        paths = [p for p in paths
                 if not (out_dir / p.stem / "report.md").exists()]
        skipped = before - len(paths)
        if skipped > 0:
            print(f"[skip_existing] skipping {skipped} already-done image(s); "
                  f"{len(paths)} remaining", flush=True)

    if args.limit > 0:
        paths = paths[:args.limit]
    return paths


# --------------------------------------------------------------------------- #
# Validation & dispatch
# --------------------------------------------------------------------------- #
def _validate_args(args):
    """Cross-check CLI argument combinations."""
    if args.regime_all:
        # Need exactly one of prompt_template / prompt_folder
        n = sum(bool(x) for x in (args.prompt_template, args.prompt_folder))
        if n == 0:
            raise SystemExit("--regime_all needs --prompt_template or --prompt_folder")
        if n > 1:
            raise SystemExit("--regime_all: --prompt_template and --prompt_folder "
                             "are mutually exclusive")
        if args.prompt_folder and not args.heatmap_dir:
            raise SystemExit("--prompt_folder requires --heatmap_dir")
        if args.prompt_template and not (args.dtd_config and args.dtd_checkpoint):
            raise SystemExit("--prompt_template requires --dtd_config + "
                             "--dtd_checkpoint")
    else:  # regime_two_stage
        if not args.adapter_stage1_path or not args.adapter_stage2_path:
            raise SystemExit("--regime_two_stage requires both "
                             "--adapter_stage1_path and --adapter_stage2_path")
        # Stage 1 prompt source
        n1 = sum(bool(x) for x in
                  (args.prompt_template_stage1, args.prompt_folder_stage1))
        if n1 != 1:
            raise SystemExit("--regime_two_stage needs exactly one of "
                             "--prompt_template_stage1 / --prompt_folder_stage1")
        # Stage 2 prompt source
        n2 = sum(bool(x) for x in
                  (args.prompt_template_stage2, args.prompt_folder_stage2))
        if n2 != 1:
            raise SystemExit("--regime_two_stage needs exactly one of "
                             "--prompt_template_stage2 / --prompt_folder_stage2")

        if args.prompt_folder_stage1 and not args.heatmap_dir:
            raise SystemExit("--prompt_folder_stage1 requires --heatmap_dir")

        # Stage 1 always needs a heatmap. If template mode -> built live with DTD.
        if args.prompt_template_stage1 and not (
            args.dtd_config and args.dtd_checkpoint
        ):
            raise SystemExit("--prompt_template_stage1 requires --dtd_config + "
                             "--dtd_checkpoint")

        # Stage-2 template needs OCR live (if stage-1 was folder mode, we run OCR
        # at stage-2 time anyway). No extra requirement.


def main() -> int:
    args = parse_args()
    _validate_args(args)

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    # Viz inputs setup
    viz_inputs_dir = None
    viz_inputs_remaining = [0]
    if args.viz_inputs:
        viz_inputs_dir = Path(args.viz_inputs).expanduser().resolve()
        viz_inputs_dir.mkdir(parents=True, exist_ok=True)
        viz_inputs_remaining = [args.viz_inputs_max_samples]
        print(f"[viz_inputs] dump up to {args.viz_inputs_max_samples} "
              f"samples to {viz_inputs_dir}")

    # Heatmap dir (for folder modes)
    heatmap_dir = None
    if args.heatmap_dir:
        heatmap_dir = Path(args.heatmap_dir).expanduser().resolve()
        if not heatmap_dir.is_dir():
            raise SystemExit(f"--heatmap_dir not a directory: {heatmap_dir}")
        n_hms = sum(1 for _ in heatmap_dir.rglob("*.png"))
        print(f"[heatmap_dir] {heatmap_dir} ({n_hms} PNGs)")

    # Gather images
    image_paths = gather_images(args)
    if not image_paths:
        print("[done] no images to process")
        return 0
    print(f"[run] {len(image_paths)} image(s)")

    # Need DTD live?
    need_dtd = False
    if args.regime_all and args.prompt_template:
        need_dtd = True
    if args.regime_two_stage and args.prompt_template_stage1:
        need_dtd = True

    dtd_bundle = None
    if need_dtd:
        print("[dtd] loading model ...")
        t0 = time.time()
        _dtd, dtd_model, dtd_model_name, dtd_needs_dct = _lazy_load_dtd(
            args.dtd_config, args.dtd_checkpoint, device,
        )
        dtd_bundle = (_dtd, dtd_model, dtd_model_name, dtd_needs_dct, device)
        print(f"[dtd] loaded {dtd_model_name} in {time.time()-t0:.1f}s")

    # ====================================================================== #
    # --regime_all
    # ====================================================================== #
    if args.regime_all:
        prompt_mode = "template" if args.prompt_template else "folder"
        prompt_template_text = None
        prompt_folder_path = None
        if prompt_mode == "template":
            prompt_template_text = Path(args.prompt_template).read_text(
                encoding="utf-8")
            print(f"[prompt] live template "
                  f"({len(prompt_template_text)} chars)")
        else:
            prompt_folder_path = Path(args.prompt_folder).expanduser().resolve()
            n = sum(1 for _ in prompt_folder_path.rglob("*.prompt.txt"))
            print(f"[prompt] folder {prompt_folder_path} ({n} prompts)")

        # Load model
        print("[student] loading ...")
        t0 = time.time()
        base, processor = _load_base_model(args, args.max_image_pixels)
        model = _attach_adapter(base, args.adapter_path)
        print(f"[student] ready in {time.time()-t0:.1f}s  "
              f"(adapter={args.adapter_path or 'NONE (zero-shot)'})")

        for i, image_path in enumerate(image_paths, start=1):
            print(f"\n[{i}/{len(image_paths)}] {image_path.name}")
            try:
                process_one_image_all(
                    image_path, args=args,
                    model=model, processor=processor,
                    dtd_bundle=dtd_bundle,
                    prompt_mode=prompt_mode,
                    prompt_template_text=prompt_template_text,
                    prompt_folder=prompt_folder_path,
                    heatmap_dir=heatmap_dir,
                    out_dir=out_dir,
                    viz_inputs_dir=viz_inputs_dir,
                    viz_inputs_remaining=viz_inputs_remaining,
                )
            except Exception as exc:
                print(f"  [error] {exc!r}")
                import traceback
                traceback.print_exc()

        print(f"\n[done] outputs in {out_dir}/")
        return 0

    # ====================================================================== #
    # --regime_two_stage
    # ====================================================================== #
    prompt_mode_s1 = "template" if args.prompt_template_stage1 else "folder"
    prompt_mode_s2 = "template" if args.prompt_template_stage2 else "folder"

    stage1_template_text = None
    stage2_template_text = None
    stage1_prompt_folder = None
    stage2_prompt_folder = None
    if prompt_mode_s1 == "template":
        stage1_template_text = Path(args.prompt_template_stage1).read_text(
            encoding="utf-8")
        print(f"[stage1 prompt] live template "
              f"({len(stage1_template_text)} chars)")
    else:
        stage1_prompt_folder = Path(args.prompt_folder_stage1
                                      ).expanduser().resolve()
        n1 = sum(1 for _ in stage1_prompt_folder.rglob("*.prompt.txt"))
        print(f"[stage1 prompt] folder {stage1_prompt_folder} ({n1})")
    if prompt_mode_s2 == "template":
        stage2_template_text = Path(args.prompt_template_stage2).read_text(
            encoding="utf-8")
        print(f"[stage2 prompt] live template "
              f"({len(stage2_template_text)} chars)")
    else:
        stage2_prompt_folder = Path(args.prompt_folder_stage2
                                      ).expanduser().resolve()
        n2 = sum(1 for _ in stage2_prompt_folder.rglob("*.prompt.txt"))
        print(f"[stage2 prompt] folder {stage2_prompt_folder} ({n2})")

    # Two adapters on the same base. To avoid loading the 32B base twice we
    # load one base, attach stage1 adapter, and at stage 2 swap the adapter
    # in-place via PeftModel's load_adapter / set_adapter mechanism. If that
    # is unavailable (e.g. very old peft), we fall back to two separate
    # PeftModel wrappers around the SAME underlying base (cheap re-wrap).
    print("[student] loading base + stage adapters ...")
    t0 = time.time()
    base, processor = _load_base_model(args, args.max_image_pixels_stage1)
    from peft import PeftModel
    stage1_model = PeftModel.from_pretrained(
        base, args.adapter_stage1_path, adapter_name="stage1",
    )
    try:
        stage1_model.load_adapter(args.adapter_stage2_path, adapter_name="stage2")
        single_model_two_adapters = True
        print(f"[student] both adapters attached to one base; "
              f"will set_adapter() to switch")
    except Exception as exc:
        print(f"[student] load_adapter for stage 2 failed ({exc!r}); "
              f"falling back to separate wrap")
        single_model_two_adapters = False

    stage1_model.set_adapter("stage1")
    stage1_model.eval()
    print(f"[student] ready in {time.time()-t0:.1f}s")

    # In single-model mode, stage2_model == stage1_model (we just switch the
    # active adapter at stage-2 time). In fallback mode, we'd build a second
    # PeftModel — but that would re-wrap the same base again, which peft
    # supports but creates an extra wrapper. For simplicity (and since the
    # standard peft we ship supports load_adapter), we expect the happy path.
    if not single_model_two_adapters:
        # Fallback: re-load base separately to host stage 2 (memory-heavy!)
        print("[student] fallback: loading second base for stage 2 "
              "(will use ~2x memory)")
        base2, processor2 = _load_base_model(args, args.max_image_pixels_stage2)
        stage2_model = PeftModel.from_pretrained(
            base2, args.adapter_stage2_path,
        )
        stage2_model.eval()
        stage2_processor = processor2
    else:
        stage2_model = stage1_model  # same object; we switch adapters
        stage2_processor = processor   # same processor (we retarget max_pixels)

    for i, image_path in enumerate(image_paths, start=1):
        print(f"\n[{i}/{len(image_paths)}] {image_path.name}")
        try:
            # Reset for this image: stage-1 adapter active, stage-1 max_pixels
            if single_model_two_adapters:
                stage1_model.set_adapter("stage1")
            _set_processor_max_pixels(processor, args.max_image_pixels_stage1)

            # Callback invoked inside process_one_image_two_stage right
            # before stage-2 generation. In single-model mode this swaps the
            # active LoRA adapter on the shared base. In fallback (separate
            # base) mode, it's a no-op.
            if single_model_two_adapters:
                def switch_to_stage2():
                    stage1_model.set_adapter("stage2")
            else:
                def switch_to_stage2():
                    pass

            process_one_image_two_stage(
                image_path, args=args,
                stage1_model=stage1_model,
                stage1_processor=processor,
                stage2_model=stage2_model,
                stage2_processor=stage2_processor,
                switch_to_stage2=switch_to_stage2,
                dtd_bundle=dtd_bundle,
                prompt_mode_stage1=prompt_mode_s1,
                prompt_mode_stage2=prompt_mode_s2,
                stage1_template_text=stage1_template_text,
                stage2_template_text=stage2_template_text,
                stage1_prompt_folder=stage1_prompt_folder,
                stage2_prompt_folder=stage2_prompt_folder,
                heatmap_dir=heatmap_dir,
                out_dir=out_dir,
                viz_inputs_dir=viz_inputs_dir,
                viz_inputs_remaining=viz_inputs_remaining,
            )
        except Exception as exc:
            print(f"  [error] {exc!r}")
            import traceback
            traceback.print_exc()

    print(f"\n[done] outputs in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())