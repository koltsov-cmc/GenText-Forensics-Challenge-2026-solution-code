#!/usr/bin/env python
"""Evaluate Stage-1 Qwen-filtered DTD output against GT masks.

Runs DTD inference + Qwen Stage 1 filtering per image, compares resulting
bboxes with GT forgery masks (SDet + SLoc). Supports metadata mode (full GT
from RealText-V2 split) and image-dir mode, with checkpointed resume.
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
from transformers import TextStreamer

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = Path(__file__).resolve().parent
_possible_roots = [_SCRIPT_DIR, _SCRIPT_DIR.parent]
_TOOLKIT_ROOT = None
for r in _possible_roots:
    if (r / "realtext_v2").is_dir():
        _TOOLKIT_ROOT = r
        break
if _TOOLKIT_ROOT is None:
    _TOOLKIT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_TOOLKIT_ROOT))

from realtext_v2.grounding import mask_to_boxes
from realtext_v2.metrics import detection_scores

# DTD
_DTD_SCRIPT_DIR = _TOOLKIT_ROOT / "ForensicHub" / "dtd_train"
sys.path.insert(0, str(_DTD_SCRIPT_DIR))
import run_doc_forensics_inference as _dtd  # noqa: E402

# OCR
sys.path.insert(0, str(_TOOLKIT_ROOT / "scripts"))
from run_paddle_sobel import run_paddle_ocr_with_lang_detect  # noqa: E402

# Optional metadata support
try:
    import pandas as pd
    from realtext_v2 import load_metadata
    from realtext_v2.metadata import resolve_paths
    _HAS_META = True
except ImportError:
    _HAS_META = False


# --------------------------------------------------------------------------- #
# Constants / helpers (from the two-stage pipeline)
# --------------------------------------------------------------------------- #
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_OUT_RE = re.compile(r"<out>(.*?)</out>", re.DOTALL | re.IGNORECASE)

_FILTERED_BBOX_RE = re.compile(
    r"\[GROUNDING\]\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)


def _extract_out(text: str) -> str:
    m = _OUT_RE.search(text)
    return m.group(1).strip() if m else ""


def _format_dtd_hints(prob: np.ndarray, threshold: float = 0.4) -> str:
    mask = (prob >= threshold).astype(np.uint8) * 255
    boxes = mask_to_boxes(mask, min_area=200)
    if not boxes:
        return "No suspicious regions detected by DTD."
    lines = [f"DTD flagged {len(boxes)} suspicious region(s):"]
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        sub = prob[y1:y2, x1:x2]
        conf = float(sub.mean()) if sub.size else 0.0
        lines.append(
            f"  Region {i}: [{x1}, {y1}, {x2}, {y2}] "
            f"(mean confidence {conf:.3f})"
        )
    return "\n".join(lines)


def _format_ocr_json(ocr_result: dict) -> str:
    payload = {
        "reading_order_text": ocr_result["reading_order_text"],
        "selected_language": ocr_result.get("selected_language", "unknown"),
        "language_scores": ocr_result.get("language_scores", {}),
        "ocr_items": [
            {"id": it["id"], "text": it["text"], "bbox": it["bbox"],
             "confidence": round(it["confidence"], 3)}
            for it in ocr_result["ocr_items"]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _boxes_to_mask(boxes: list[list[int]], h: int, w: int) -> np.ndarray:
    """Convert list of [x1,y1,x2,y2] to a binary mask of shape (h, w)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def _pixel_iou_f1(gt_mask: np.ndarray, pred_mask: np.ndarray
                  ) -> tuple[float, float]:
    gt = (gt_mask > 0).astype(np.uint8)
    pr = (pred_mask > 0).astype(np.uint8)
    inter = int(np.logical_and(gt, pr).sum())
    union = int(np.logical_or(gt, pr).sum())
    iou = (inter / union) if union > 0 else float(gt.sum() == 0 and pr.sum() == 0)
    tp = inter
    fp = int(np.logical_and(pr == 1, gt == 0).sum())
    fn = int(np.logical_and(pr == 0, gt == 1).sum())
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else float(gt.sum() == 0 and pr.sum() == 0)
    return float(iou), float(f1)


# --------------------------------------------------------------------------- #
# Qwen generation
# --------------------------------------------------------------------------- #
def _qwen_generate(qwen_model, processor, messages, gen_kwargs) -> str:
    import torch
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: (v.to(qwen_model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    with torch.inference_mode():
        streamer = TextStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        out_ids = qwen_model.generate(**inputs, **gen_kwargs, streamer=streamer)
    trimmed = [o[len(iid):] for iid, o in zip(inputs["input_ids"], out_ids)]
    texts = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return texts[0]



# --------------------------------------------------------------------------- #
# Checkpoint load / save
# --------------------------------------------------------------------------- #
def _load_checkpoint(ckpt_path: Path) -> dict | None:
    if not ckpt_path.exists():
        return None
    try:
        return json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_checkpoint(ckpt_path: Path, **kwargs) -> None:
    ckpt_path.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2),
                         encoding="utf-8")


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _save_visualization(
    image_pil: Image.Image,
    prob: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    out_path: Path,
    *,
    sample_id: str,
    is_forged_gt: Optional[bool],
    pred_forged: bool,
    iou: Optional[float],
    f1: Optional[float],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_arr = np.asarray(image_pil.convert("RGB"))
    H, W = image_arr.shape[:2]

    panel_w_in = min(5.5, max(3.0, W / 350))
    panel_h_in = panel_w_in * H / W
    fig_w = panel_w_in * 4 + 0.6
    fig_h = panel_h_in + 1.2
    fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h))

    # Panel 1 — image
    axes[0].imshow(image_arr)
    axes[0].set_title("image", fontsize=10)
    axes[0].axis("off")

    # Panel 2 — GT mask
    if gt_mask is not None:
        axes[1].imshow(image_arr, alpha=0.35)
        axes[1].imshow(gt_mask, cmap="Reds", alpha=0.65, vmin=0, vmax=255)
        t = "GT mask"
        if is_forged_gt is not None:
            t += f"  ({'FORGED' if is_forged_gt else 'AUTHENTIC'})"
    else:
        axes[1].imshow(image_arr)
        axes[1].text(0.5, 0.5, "no GT mask", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=12, color="#cccccc",
                     bbox=dict(facecolor="black", alpha=0.55, pad=8))
        t = "GT mask (n/a)"
    axes[1].set_title(t, fontsize=10)
    axes[1].axis("off")

    # Panel 3 — DTD overlay
    cmap = plt.get_cmap("jet")
    heat = (cmap(prob)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * image_arr + 0.45 * heat).clip(0, 255).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"DTD prob   max={prob.max():.3f}  mean={prob.mean():.3f}", fontsize=10,
    )
    axes[2].axis("off")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = axes[2].inset_axes([0.05, -0.08, 0.9, 0.04])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7)

    # Panel 4 — Qwen-filtered mask
    axes[3].imshow(image_arr, alpha=0.35)
    axes[3].imshow(pred_mask, cmap="Greens", alpha=0.6, vmin=0, vmax=255)
    p4 = (f"Qwen stage-1 mask   "
          f"verdict={'FORGED' if pred_forged else 'AUTHENTIC'}")
    if iou is not None and f1 is not None:
        p4 += f"\nIoU={iou:.3f}  F1={f1:.3f}"
    axes[3].set_title(p4, fontsize=10)
    axes[3].axis("off")

    fig.suptitle(sample_id, fontsize=11, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# GT mask finder
# --------------------------------------------------------------------------- #
def _find_gt_mask(masks_dir: Path | None, stem: str) -> Path | None:
    if masks_dir is None or not masks_dir.is_dir():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        for cand in (masks_dir / f"{stem}{ext}",
                     masks_dir / f"{stem}_mask{ext}"):
            if cand.exists():
                return cand
    return None


# --------------------------------------------------------------------------- #
# Sample-list builders
# --------------------------------------------------------------------------- #
def _build_samples_from_metadata(args) -> list[dict]:
    if not _HAS_META:
        raise SystemExit("metadata mode requires pandas + realtext_v2 installed.")
    if not args.root:
        raise SystemExit("metadata mode requires --root.")
    print("[data] loading metadata ...")
    meta = load_metadata(args.root)
    meta = resolve_paths(meta, args.root)
    if args.split_parquet:
        split_df = pd.read_parquet(args.split_parquet)
        keep_ids = set(split_df.get(
            "original_sample_id", split_df.get("sample_id", [])
        ).tolist())
        meta = meta[meta["sample_id"].isin(keep_ids)].reset_index(drop=True)
        print(f"[data] split restricted to {len(meta)} rows")
    if args.limit > 0:
        n_black = (meta["type"] == "black").sum()
        n_white = (meta["type"] == "white").sum()
        forged = meta[meta["type"] == "black"].sample(
            n=min(args.limit // 2, n_black), random_state=args.seed,
        )
        pristine = meta[meta["type"] == "white"].sample(
            n=min(args.limit - len(forged), n_white), random_state=args.seed,
        )
        meta = pd.concat([forged, pristine]).reset_index(drop=True)
    samples = []
    for _, row in meta.iterrows():
        img_path = row.get("image_path")
        mask_path = row.get("mask_path")
        if img_path is None:
            continue
        gt_type = str(row.get("type", "")).lower()
        samples.append({
            "sample_id":    str(row["sample_id"]),
            "image_path":   Path(str(img_path)),
            "mask_path":    Path(str(mask_path)) if mask_path else None,
            "is_forged_gt": gt_type.startswith("black"),
        })
    return samples


def _build_samples_from_dir(args) -> list[dict]:
    img_dir = Path(args.image_dir).expanduser().resolve()
    if not img_dir.is_dir():
        raise SystemExit(f"--image_dir not found: {img_dir}")
    masks_dir = (Path(args.gt_masks_dir).expanduser().resolve()
                 if args.gt_masks_dir else None)
    paths = sorted(p for p in img_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if args.order == "random":
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(paths))
        paths = [paths[i] for i in idx]
    if args.limit > 0:
        paths = paths[:args.limit]
    samples = []
    for p in paths:
        stem = p.stem
        gt_path = _find_gt_mask(masks_dir, stem)
        is_forged_gt: Optional[bool] = None
        if gt_path is not None:
            try:
                arr = np.array(Image.open(str(gt_path)).convert("L"),
                               dtype=np.uint8)
                is_forged_gt = bool((arr > 0).any())
            except Exception:
                is_forged_gt = False
        else:
            is_forged_gt = False
        samples.append({
            "sample_id":    stem,
            "image_path":   p,
            "mask_path":    gt_path,
            "is_forged_gt": is_forged_gt,
        })
    return samples


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # DTD
    ap.add_argument("--config", required=True, help="DTD YAML config.")
    ap.add_argument("--checkpoint", required=True, help="DTD checkpoint .pth.")

    # Mode A
    g_meta = ap.add_argument_group("metadata mode")
    g_meta.add_argument("--root", help="RealText-V2 root.")
    g_meta.add_argument("--split_parquet", help="Split parquet.")

    # Mode B
    g_dir = ap.add_argument_group("image-dir mode")
    g_dir.add_argument("--image_dir", help="Directory of images.")
    g_dir.add_argument("--gt_masks_dir", help="Directory of GT masks.")
    g_dir.add_argument("--order", choices=("sequential", "random"),
                       default="sequential")
    g_dir.add_argument("--seed", type=int, default=42)

    # Qwen
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--model_class", default="Qwen3VLForConditionalGeneration")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--attn_impl", default="sdpa",
                    choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--greedy", action="store_true")

    # Common
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtd_threshold", type=float, default=0.4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--langs", default="en,ch,th,ms,id,ar")
    ap.add_argument("--out_dir", default="eval/stage1")
    ap.add_argument("--save_viz", action="store_true")
    ap.add_argument("--no_resume", action="store_true")

    ap.add_argument("--prompt", default="prompt.txt")

    # TTA
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tta_combine", default="min", choices=("min", "mean", "median"))
    ap.add_argument("--tta_passes", type=int, default=4, choices=(3, 4))
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    import torch
    import transformers
    from transformers import AutoProcessor

    if bool(args.image_dir) == bool(args.root):
        raise SystemExit(
            "Specify exactly one of --image_dir or --root."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # ---- Sample list ----
    if args.image_dir:
        samples = _build_samples_from_dir(args)
        mode = "image-dir"
    else:
        samples = _build_samples_from_metadata(args)
        mode = "metadata"
    print(f"[data] mode={mode}  {len(samples)} samples")

    if not samples:
        print("[error] no samples")
        return 1

    # ---- Load DTD ----
    print("[dtd] loading model ...")
    t0 = time.time()
    _dtd._setup_paths_and_registry()
    dtd_model, dtd_model_name, dtd_needs_dct = _dtd.build_model_and_load(
        args.config, args.checkpoint, device,
    )
    print(f"[dtd] loaded {dtd_model_name} in {time.time()-t0:.1f}s")

    # ---- Load Qwen ----
    print("[qwen] loading model ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    ModelCls = getattr(transformers, args.model_class)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    qwen_model = ModelCls.from_pretrained(
        args.model_id, dtype=dtype_map[args.dtype],
        device_map=args.device_map, attn_implementation=args.attn_impl,
        trust_remote_code=True,
    ).eval()
    print(f"[qwen] loaded in {time.time()-t0:.1f}s")

    # ---- Checkpoint ----
    ckpt_path = out_dir / "progress.json"
    if args.no_resume:
        ckpt = None
    else:
        ckpt = _load_checkpoint(ckpt_path)

    if ckpt:
        completed_ids = set(ckpt.get("completed_ids", []))
        y_true         = list(ckpt.get("y_true", []))
        y_pred         = list(ckpt.get("y_pred", []))
        n_forged_gt    = int(ckpt.get("n_forged_gt", 0))
        n_forged_pred  = int(ckpt.get("n_forged_pred", 0))
        n_no_gt        = int(ckpt.get("n_no_gt", 0))
        grounding_ious = list(ckpt.get("grounding_iou", []))
        grounding_f1s  = list(ckpt.get("grounding_f1", []))
        print(f"[resume] {len(completed_ids)} done, "
              f"{len(y_true)} with GT, {len(grounding_ious)} grounded")
    else:
        completed_ids = set()
        y_true, y_pred = [], []
        n_forged_gt, n_forged_pred, n_no_gt = 0, 0, 0
        grounding_ious: list[float] = []
        grounding_f1s:  list[float] = []

    n_total = len(samples)
    pending = [s for s in samples if s["sample_id"] not in completed_ids]
    if len(pending) < len(samples):
        print(f"[resume] {len(pending)} remaining of {n_total}")

    # Gen kwargs (constant across all images)
    do_sample = not args.greedy
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=processor.tokenizer.pad_token_id
        or processor.tokenizer.eos_token_id,
    )

    # ---- Evaluation loop ----
    for s in pending:
        img_path = s["image_path"]
        mask_path = s["mask_path"]
        sample_id = s["sample_id"]
        is_forged_gt = s["is_forged_gt"]

        if not img_path.exists():
            print(f"  [{len(completed_ids)+1}/{n_total}] {sample_id}: SKIP")
            completed_ids.add(sample_id)
            _save_checkpoint(
                ckpt_path,
                n_total=n_total, completed_ids=sorted(completed_ids),
                y_true=y_true, y_pred=y_pred,
                n_forged_gt=n_forged_gt, n_forged_pred=n_forged_pred,
                n_no_gt=n_no_gt,
                grounding_iou=grounding_ious, grounding_f1=grounding_f1s,
            )
            continue

        gt_str = (f"forged_gt={is_forged_gt}" if is_forged_gt is not None
                  else "forged_gt=?")
        print(f"\n[{len(completed_ids)+1}/{n_total}] {sample_id}  {gt_str}")

        # -- DTD inference --
        tta_offsets = _dtd._build_tta_offsets(args.tta, args.tta_passes)
        if tta_offsets is not None:
            print(f"  [tta] passes={len(tta_offsets)}  "
                  f"combine={args.tta_combine}")
        t0 = time.time()
        prob, image_pil = _dtd.infer_one_image(
            img_path, dtd_model, dtd_model_name, dtd_needs_dct, device,
            jpeg_quality=args.jpeg_quality,
            tta_offsets=tta_offsets,
            tta_combine=args.tta_combine,
        )
        dtd_elapsed = time.time() - t0
        print(f"  [dtd] {dtd_elapsed:.1f}s  max={float(prob.max()):.4f}  "
              f"mean={float(prob.mean()):.4f}")

        orig_w, orig_h = image_pil.size
        img_arr = np.asarray(image_pil.convert("RGB"))

        # -- DTD heatmap overlay (for Qwen) --
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("jet")
        heat = (cmap(prob)[:, :, :3] * 255).astype(np.uint8)
        dtd_overlay_arr = (0.55 * img_arr + 0.45 * heat).clip(0, 255).astype(np.uint8)
        dtd_overlay_pil = Image.fromarray(dtd_overlay_arr)

        # -- OCR --
        ocr_result = run_paddle_ocr_with_lang_detect(
            img_path,
            candidate_langs=[s.strip() for s in args.langs.split(",") if s.strip()],
            gpu=torch.cuda.is_available(),
            mag_ratio=1.0,
            verbose=False,
        )
        ocr_json_str = _format_ocr_json(ocr_result)
        dtd_hints_str = _format_dtd_hints(prob, threshold=args.dtd_threshold)

        # -- Stage 1: Qwen filtering --
        stage1_prompt_path = _TOOLKIT_ROOT / "prompts" / args.prompt
        stage1_prompt = (
            stage1_prompt_path.read_text(encoding="utf-8")
            .replace("{{OCR_JSON}}", ocr_json_str)
        )
        stage1_prompt += f"\n\nDTD DETECTOR OUTPUT:\n{dtd_hints_str}"
        stage1_prompt += (
            f"\n\nIMAGE METADATA: Width={orig_w} Height={orig_h}. "
            f"All coordinates are absolute integer pixels."
        )

        stage1_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "image", "image": dtd_overlay_pil},
                {"type": "text", "text": stage1_prompt},
            ],
        }]

        print("  [qwen] stage 1 filtering ...", flush=True)
        t0 = time.time()
        stage1_raw = _qwen_generate(qwen_model, processor, stage1_messages, gen_kwargs)
        stage1_elapsed = time.time() - t0
        print(f"\n  [qwen] done in {stage1_elapsed:.1f}s", flush=True)

        # -- Extract filtered bboxes --
        filtered_text = _extract_out(stage1_raw)
        filtered_boxes: list[list[int]] = []
        for m in _FILTERED_BBOX_RE.finditer(filtered_text):
            filtered_boxes.append([int(m.group(j)) for j in range(1, 5)])
        n_filtered = len(filtered_boxes)
        print(f"  [filter] {n_filtered} bbox(es) kept after Qwen filtering")

        # -- Detection verdict --
        pred_forged = n_filtered > 0
        if pred_forged:
            n_forged_pred += 1
        if is_forged_gt is True:
            n_forged_gt += 1

        if is_forged_gt is None:
            n_no_gt += 1
            print(f"  [det] gt=?  pred={'FORGED' if pred_forged else 'AUTHENTIC'}")
        else:
            gt_label = "FORGED" if is_forged_gt else "AUTHENTIC"
            pred_label = "FORGED" if pred_forged else "AUTHENTIC"
            y_true.append(gt_label)
            y_pred.append(pred_label)
            print(f"  [det] gt={gt_label}  pred={pred_label}")

        # -- GT mask --
        gt_mask: Optional[np.ndarray] = None
        if mask_path is not None and Path(str(mask_path)).exists():
            m = Image.open(str(mask_path)).convert("L")
            if m.size != (orig_w, orig_h):
                m = m.resize((orig_w, orig_h), Image.NEAREST)
            gt_mask = np.array(m, dtype=np.uint8)
            gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        # -- Pred mask from filtered bboxes --
        pred_mask = _boxes_to_mask(filtered_boxes, orig_h, orig_w)

        # -- IoU / F1 --
        iou_i: Optional[float] = None
        f1_i: Optional[float] = None
        if gt_mask is not None and is_forged_gt is not None:
            iou_i, f1_i = _pixel_iou_f1(gt_mask, pred_mask)
            grounding_ious.append(iou_i)
            grounding_f1s.append(f1_i)
            print(f"  [loc] iou={iou_i:.4f}  f1={f1_i:.4f}")

        # -- Visualization --
        if args.save_viz:
            _save_visualization(
                image_pil=image_pil, prob=prob,
                gt_mask=gt_mask, pred_mask=pred_mask,
                out_path=viz_dir / f"{sample_id}.viz.png",
                sample_id=sample_id,
                is_forged_gt=is_forged_gt,
                pred_forged=pred_forged,
                iou=iou_i, f1=f1_i,
            )

        # -- Save raw stage1 output --
        (out_dir / f"{sample_id}.stage1_raw.txt").write_text(
            stage1_raw, encoding="utf-8",
        )
        if filtered_text:
            (out_dir / f"{sample_id}.stage1_filtered.txt").write_text(
                filtered_text, encoding="utf-8",
            )

        # -- Checkpoint --
        completed_ids.add(sample_id)
        _save_checkpoint(
            ckpt_path,
            n_total=n_total, completed_ids=sorted(completed_ids),
            y_true=y_true, y_pred=y_pred,
            n_forged_gt=n_forged_gt, n_forged_pred=n_forged_pred,
            n_no_gt=n_no_gt,
            grounding_iou=grounding_ious, grounding_f1=grounding_f1s,
        )

    # ---- Aggregate metrics ----
    print("\n" + "=" * 60)
    print("METRICS (Qwen Stage 1 filtering)")
    print("=" * 60)

    out_dict: dict = {
        "mode": mode,
        "n_total": int(n_total),
        "n_with_gt": len(y_true),
        "n_no_gt": n_no_gt,
        "dtd_threshold": args.dtd_threshold,
        "forged_gt_count": n_forged_gt,
        "forged_pred_count": n_forged_pred,
    }

    if y_true:
        det = detection_scores(y_true, y_pred)
        print(f"\n[Detection (SDet)]")
        print(f"  precision = {det['precision']:.4f}")
        print(f"  recall    = {det['recall']:.4f}")
        print(f"  f1        = {det['f1']:.4f}")
        print(f"  accuracy  = {det['accuracy']:.4f}")
        print(f"  n         = {det['n']}")
        print(f"  forged_gt = {n_forged_gt}  forged_pred = {n_forged_pred}")
        out_dict["detection"] = det
    else:
        print("\n[Detection] skipped — no GT labels.")

    if grounding_ious:
        ious_arr = np.array(grounding_ious, dtype=np.float64)
        f1s_arr  = np.array(grounding_f1s,  dtype=np.float64)
        mIoU = float(ious_arr.mean())
        mF1  = float(f1s_arr.mean())
        loc = {
            "mIoU": mIoU, "mF1": mF1,
            "mIoU_forged_only": mIoU, "mF1_forged_only": mF1,
            "n": len(grounding_ious), "n_forged": n_forged_gt,
        }
        print(f"\n[Grounding (SLoc)]")
        print(f"  mIoU            = {mIoU:.4f}")
        print(f"  mF1             = {mF1:.4f}")
        print(f"  n               = {loc['n']}  (forged={n_forged_gt})")
        s_loc = 0.5 * mIoU + 0.5 * mF1
        print(f"\n  SLoc composite = {s_loc:.4f}")
        out_dict["grounding"] = loc
        out_dict["s_loc_composite"] = float(s_loc)
    else:
        print("\n[Grounding] skipped — no GT masks.")

    # ---- Save final results ----
    out_path = out_dir / "eval.json"
    out_path.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
    print(f"\n[done] results -> {out_path}")

    if ckpt_path.exists():
        ckpt_path.unlink()
        print("[resume] progress checkpoint removed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
