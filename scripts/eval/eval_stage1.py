#!/usr/bin/env python
"""Evaluate Stage-1 Qwen filtering: compare raw DTD vs filtered bboxes against GT mask.

Loads the Stage-1 LoRA adapter, extracts Qwen-kept bboxes, and computes
pixel-level IoU with GT forgery masks. Supports live prompt construction
and pre-rendered prompt files.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Path setup (identical to train_student_sft.py)
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = Path(__file__).resolve().parent
for r in (_SCRIPT_DIR, _SCRIPT_DIR.parent):
    if (r / "realtext_v2").is_dir() or (r / "ForensicHub").is_dir():
        _TOOLKIT_ROOT = r
        break
else:
    _TOOLKIT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_TOOLKIT_ROOT))

from realtext_v2.grounding import mask_to_boxes as _toolkit_mask_to_boxes  # noqa: E402

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# --------------------------------------------------------------------------- #
# Helpers copied / adapted from train_student_sft.py
# --------------------------------------------------------------------------- #
def _smart_resize(w: int, h: int, max_pixels: int) -> tuple[int, int]:
    """Approximate Qwen3-VL smart_resize."""
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


def _format_dtd_hints(dtd_regions: list) -> str:
    if not dtd_regions:
        return "No suspicious regions detected by DTD."
    lines = [f"DTD flagged {len(dtd_regions)} suspicious region(s):"]
    for i, box in enumerate(dtd_regions, start=1):
        x1, y1, x2, y2 = (int(v) for v in box)
        lines.append(f"  Region {i}: [{x1}, {y1}, {x2}, {y2}]")
    return "\n".join(lines)


def _find_heatmap_path(heatmap_dir: Path, part: str, stem: str) -> Optional[Path]:
    candidates = [
        heatmap_dir / part / f"{stem}.heatmap.png",
        heatmap_dir / part / f"{stem}_dtd.png",
        heatmap_dir / part / f"{stem}.dtd.png",
        heatmap_dir / part / f"{stem}_heatmap.png",
        heatmap_dir / f"{stem}_dtd.png",
        heatmap_dir / f"{stem}.heatmap.png",
        heatmap_dir / f"{stem}.dtd.png",
        heatmap_dir / f"{stem}_heatmap.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_image_path(image_root: Path, part: str, stem: str) -> Optional[Path]:
    part_dir = image_root / part
    if not part_dir.is_dir():
        return None
    for ext in _IMG_EXTS:
        p = part_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _find_gt_mask(mask_dir: Path, part: str, stem: str) -> Optional[Path]:
    part_dir = mask_dir / part
    if not part_dir.is_dir():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        p = part_dir / f"{stem}_mask{ext}"
        if p.exists():
            return p
        p = part_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _find_prerendered_prompt(prompt_dir: Path, part: str, stem: str) -> Optional[Path]:
    """Look for pre-rendered prompt text file."""
    candidates = [
        prompt_dir / part / f"{stem}_prompt.txt",
        prompt_dir / part / f"{stem}.txt",
        prompt_dir / part / f"{stem}_input_prompt.txt",
        prompt_dir / f"{stem}_prompt.txt",
        prompt_dir / f"{stem}.txt",
        prompt_dir / f"{stem}_input_prompt.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- #
# BBox extraction from Qwen raw output
# --------------------------------------------------------------------------- #
_BBOX_RE = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
)


def _extract_bboxes_from_text(text: str) -> list[tuple[int, int, int, int]]:
    """Return every [x1,y1,x2,y2] found in the text."""
    boxes = []
    for m in _BBOX_RE.finditer(text):
        boxes.append((int(m.group(1)), int(m.group(2)),
                      int(m.group(3)), int(m.group(4))))
    seen = set()
    out = []
    for b in boxes:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _extract_kept_bboxes(text: str) -> list[tuple[int, int, int, int]]:
    """Parse Stage-1 output and keep only bboxes from REGION blocks that end
    with a KEEP line (not DROP)."""
    blocks = re.split(r"(?=REGION\s+\d+)", text)
    kept = []
    for block in blocks:
        block = block.strip()
        if not block.startswith("REGION"):
            continue
        if re.search(
            r"KEEP\s+as\s+(?:Semantic\s+Subtle|Visual\s+Clumsy|Logical\s+Fraud)",
            block, re.IGNORECASE,
        ):
            kept.extend(_extract_bboxes_from_text(block))
    return kept


# --------------------------------------------------------------------------- #
# IoU helpers
# --------------------------------------------------------------------------- #
def _boxes_to_mask(boxes: list[tuple[int, int, int, int]], h: int, w: int) -> np.ndarray:
    """Paint a binary mask from a list of bboxes."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def _mask_to_bboxes(mask_path: Path, min_area: int = 10) -> list[tuple[int, int, int, int]]:
    m = Image.open(str(mask_path)).convert("L")
    arr = np.array(m, dtype=np.uint8)
    binary = (arr > 0).astype(np.uint8)
    if not binary.any():
        return []
    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n_comp):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Model loading (identical logic to train_student_sft.py)
# --------------------------------------------------------------------------- #
def _load_model(checkpoint_dir: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, AutoModelForVision2Seq

    print("[model] loading processor ...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-32B-Instruct", trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[model] loading base model ...")
    base = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen3-VL-32B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[model] loading LoRA adapter from {checkpoint_dir} ...")
    model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    model.eval()
    print("[model] ready")
    return model, processor


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _infer_one(
    image_pil: Image.Image,
    heatmap_pil: Image.Image,
    prompt_text: str,
    model,
    processor,
    max_new_tokens: int = 8192,
) -> str:
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "image", "image": heatmap_pil},
            {"type": "text", "text": prompt_text},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=(processor.tokenizer.pad_token_id
                          or processor.tokenizer.eos_token_id),
        )

    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    return text


# --------------------------------------------------------------------------- #
# Viz helper
# --------------------------------------------------------------------------- #
def _save_viz(
    viz_dir: Path,
    stem: str,
    image_pil: Image.Image,
    heatmap_pil: Image.Image,
    prompt: str,
    output: str,
):
    """Save resized image, heatmap, prompt, and model output for inspection."""
    sample_dir = viz_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_pil.save(sample_dir / f"{stem}_input_image.png")
    heatmap_pil.save(sample_dir / f"{stem}_input_heatmap.png")
    (sample_dir / f"{stem}_input_prompt.txt").write_text(prompt, encoding="utf-8")
    (sample_dir / f"{stem}_output.txt").write_text(output, encoding="utf-8")
    meta = {
        "processed_size": [image_pil.size[0], image_pil.size[1]],
    }
    (sample_dir / f"{stem}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cot_root", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--heatmap_dir", required=True)
    ap.add_argument("--gt_mask_dir", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to LoRA adapter directory (e.g. runs/.../adapter_final)")
    ap.add_argument("--stage1_prompt", default=None,
                    help="Template for prompt building (also used for viz label). "
                         "Required unless --prerendered_prompts_dir is set.")
    ap.add_argument("--prerendered_prompts_dir", default=None,
                    help="If set, read pre-built prompt text from this dir "
                         "instead of building {{DTD_HINTS}} on the fly.")
    ap.add_argument("--out_dir", default="eval/stage1_filtering")
    ap.add_argument("--viz_inputs", default=None,
                    help="If set, save per-sample viz artifacts here.")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--max_image_pixels", type=int, default=0,
                    help="If >0, resize image+heatmap via smart_resize "
                         "before feeding to model (default: native resolution).")
    args = ap.parse_args()

    cot_root = Path(args.cot_root).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    heatmap_dir = Path(args.heatmap_dir).expanduser().resolve()
    gt_mask_dir = Path(args.gt_mask_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = (Path(args.viz_inputs).expanduser().resolve()
               if args.viz_inputs else None)
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"[viz] saving inputs to {viz_dir}")

    prompt_dir = (Path(args.prerendered_prompts_dir).expanduser().resolve()
                  if args.prerendered_prompts_dir else None)
    if prompt_dir is not None:
        print(f"[prompts] using pre-rendered prompts from {prompt_dir}")

    if prompt_dir is None and args.stage1_prompt is None:
        raise SystemExit(
            "Either --stage1_prompt or --prerendered_prompts_dir must be provided."
        )

    # Load prompt template (needed even with prerendered for viz label)
    stage1_template = ""
    if args.stage1_prompt is not None:
        prompt_path = Path(args.stage1_prompt).expanduser().resolve()
        stage1_template = prompt_path.read_text(encoding="utf-8")

    # Load model
    model, processor = _load_model(Path(args.checkpoint).expanduser().resolve())

    # Gather JSON files
    cot_files: list[tuple[Path, str]] = []
    for part_idx in range(20):
        part = f"part{part_idx:03d}"
        part_dir = cot_root / part
        if not part_dir.is_dir():
            continue
        for p in sorted(part_dir.glob("*.cot.json")):
            cot_files.append((p, part))

    if args.limit > 0:
        cot_files = cot_files[:args.limit]

    print(f"[eval] {len(cot_files)} sample(s)")

    results = []
    for idx, (cot_path, part) in enumerate(cot_files, start=1):
        stem = cot_path.stem
        if stem.endswith(".cot"):
            stem = stem[:-4]

        print(f"\n[{idx}/{len(cot_files)}] {stem}")

        # --- Load JSON ---
        rec = json.loads(cot_path.read_text(encoding="utf-8"))
        dtd_regions = rec.get("dtd_regions", [])

        # --- Resolve paths ---
        img_path = _find_image_path(image_root, part, stem)
        hm_path = _find_heatmap_path(heatmap_dir, part, stem)
        mask_path = _find_gt_mask(gt_mask_dir, part, stem)

        if img_path is None or hm_path is None:
            print(f"  [skip] missing image or heatmap")
            continue

        # --- Load images ---
        image_pil = Image.open(img_path).convert("RGB")
        heatmap_pil = Image.open(hm_path).convert("RGB")
        orig_w, orig_h = image_pil.size

        # --- smart_resize if requested ---
        if args.max_image_pixels > 0:
            proc_w, proc_h = _smart_resize(orig_w, orig_h, args.max_image_pixels)
            if (proc_w, proc_h) != (orig_w, orig_h):
                image_pil = image_pil.resize((proc_w, proc_h), Image.BILINEAR)
                heatmap_pil = heatmap_pil.resize((proc_w, proc_h), Image.BILINEAR)
        else:
            # Just match heatmap to image (legacy behaviour)
            if heatmap_pil.size != image_pil.size:
                heatmap_pil = heatmap_pil.resize(image_pil.size, Image.BILINEAR)

        # --- Build or load prompt ---
        if prompt_dir is not None:
            pr_path = _find_prerendered_prompt(prompt_dir, part, stem)
            if pr_path is None:
                print(f"  [skip] no pre-rendered prompt for {stem}")
                continue
            prompt = pr_path.read_text(encoding="utf-8")
        else:
            dtd_text = _format_dtd_hints(dtd_regions)
            prompt = stage1_template.replace("{{DTD_HINTS}}", dtd_text)

        # --- Inference ---
        t0 = time.time()
        raw_output = _infer_one(
            image_pil, heatmap_pil, prompt,
            model, processor, max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - t0
        print(f"  [qwen] {elapsed:.1f}s  chars={len(raw_output)}")

        # --- Extract bboxes ---
        qwen_all_boxes = _extract_bboxes_from_text(raw_output)
        qwen_kept_boxes = _extract_kept_bboxes(raw_output)

        # --- IoU computation ---
        h, w = image_pil.size[1], image_pil.size[0]
        gt_mask = _mask_to_bboxes(mask_path) if mask_path else []
        gt_mask_arr = _boxes_to_mask(gt_mask, h, w) if gt_mask else np.zeros((h, w), dtype=np.uint8)

        dtd_mask = _boxes_to_mask(
            [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in dtd_regions],
            h, w,
        )
        qwen_mask = _boxes_to_mask(qwen_kept_boxes, h, w)

        iou_dtd = _compute_iou(dtd_mask, gt_mask_arr)
        iou_qwen = _compute_iou(qwen_mask, gt_mask_arr)

        print(f"  [iou]  raw_dtd={iou_dtd:.3f}  qwen_kept={iou_qwen:.3f}  "
              f"n_dtd={len(dtd_regions)}  n_kept={len(qwen_kept_boxes)}")

        # --- Save per-sample ---
        sample_out = {
            "stem": stem,
            "part": part,
            "raw_output": raw_output,
            "dtd_regions": dtd_regions,
            "qwen_all_boxes": qwen_all_boxes,
            "qwen_kept_boxes": qwen_kept_boxes,
            "iou_raw_dtd_vs_gt": iou_dtd,
            "iou_qwen_kept_vs_gt": iou_qwen,
            "n_dtd": len(dtd_regions),
            "n_qwen_kept": len(qwen_kept_boxes),
            "elapsed_sec": elapsed,
        }
        results.append(sample_out)
        (out_dir / f"{stem}.eval.json").write_text(
            json.dumps(sample_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # --- Viz ---
        if viz_dir is not None:
            _save_viz(viz_dir, stem, image_pil, heatmap_pil, prompt, raw_output)

    # --- Summary ---
    if results:
        avg_dtd = sum(r["iou_raw_dtd_vs_gt"] for r in results) / len(results)
        avg_qwen = sum(r["iou_qwen_kept_vs_gt"] for r in results) / len(results)
        summary = {
            "n_samples": len(results),
            "avg_iou_raw_dtd": avg_dtd,
            "avg_iou_qwen_kept": avg_qwen,
            "improvement": avg_qwen - avg_dtd,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[summary] n={len(results)}  raw_dtd={avg_dtd:.3f}  "
              f"qwen_kept={avg_qwen:.3f}  delta={avg_qwen-avg_dtd:+.3f}")
    else:
        print("[summary] no valid samples processed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
