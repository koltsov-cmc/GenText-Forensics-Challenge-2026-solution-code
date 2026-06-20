#!/usr/bin/env python
"""Two-stage Qwen pipeline: DTD region filtering (Stage 1) then final forensic
report generation (Stage 2).

Stage 1 validates DTD regions using image + DTD heatmap + bounding boxes.
Stage 2 generates the full report using the original image, OCR output, and
filtered anomalies from Stage 1.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import TextStreamer

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = Path(__file__).resolve().parent
_possible_roots = [_SCRIPT_DIR, _SCRIPT_DIR.parent.parent]
_TOOLKIT_ROOT = None
for r in _possible_roots:
    if (r / "realtext_v2").is_dir() or (r / "models").is_dir():
        _TOOLKIT_ROOT = r
        break
if _TOOLKIT_ROOT is None:
    _TOOLKIT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_TOOLKIT_ROOT))

from realtext_v2.grounding import mask_to_boxes
from realtext_v2.report import parse_report, ForgeryReport

# DTD
_DTD_SCRIPT_DIR = _TOOLKIT_ROOT / "ForensicHub" / "dtd_train"
sys.path.insert(0, str(_DTD_SCRIPT_DIR))
import run_doc_forensics_inference as _dtd  # noqa: E402

# OCR + viz
sys.path.insert(0, str(_TOOLKIT_ROOT / "scripts"))
from run_paddle_sobel import run_paddle_ocr_with_lang_detect, draw_ocr_boxes  # noqa: E402
from vis_report import visualize_report                                     # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers (same as single-stage pipeline)
# --------------------------------------------------------------------------- #
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


_REPORT_ANCHOR_RE = re.compile(r"#\s*FORGERY\s+ANALYSIS\s+REPORT", re.IGNORECASE)
_END_MARKER = "**END OF REPORT**"
_OUT_RE = re.compile(r"<out>(.*?)</out>", re.DOTALL | re.IGNORECASE)
_REPORT_RE = re.compile(r"<report>(.*?)</report>", re.DOTALL | re.IGNORECASE)


def _extract_out(text: str) -> str:
    """Extract the content between <out>...</out> from stage 1 output."""
    m = _OUT_RE.search(text)
    return m.group(1).strip() if m else ""


def _extract_clean_report(text: str) -> str:
    """Extract the final report from stage 2 output (inside <report> or
    anchored by '# FORGERY ANALYSIS REPORT' with END marker)."""
    # Prefer <report> block
    m = _REPORT_RE.search(text)
    if m:
        report = m.group(1).strip()
    else:
        report = text

    # Strip fences
    report = re.sub(r"^```(?:markdown|md)?\s*", "", report)
    report = re.sub(r"\s*```$", "", report)
    report = report.strip()

    # Anchor to last '# FORGERY ANALYSIS REPORT'
    anchors = list(_REPORT_ANCHOR_RE.finditer(report))
    if anchors:
        report = report[anchors[-1].start():]

    # Cut at END OF REPORT
    end_idx = report.find(_END_MARKER)
    if end_idx >= 0:
        report = report[:end_idx + len(_END_MARKER)]

    return report.strip()


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
# Qwen generation helper
# --------------------------------------------------------------------------- #
def _qwen_generate(
    qwen_model,
    processor,
    messages: list[dict],
    gen_kwargs: dict,
) -> str:
    """Run generation with TextStreamer, return decoded text."""
    import torch

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
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


_FILTERED_BBOX_RE = re.compile(
    r"\[GROUNDING\]\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)


def _draw_filtered_boxes(
    image: Image.Image,
    filtered_text: str,
    out_path: Path,
    title: str = "",
) -> None:
    """Draw bounding boxes from Stage 1 filtered anomalies on the image."""
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

    boxes = []
    for m in _FILTERED_BBOX_RE.finditer(filtered_text):
        boxes.append([int(m.group(j)) for j in range(1, 5)])

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        rect = mpatches.Rectangle(
            (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
            linewidth=2, edgecolor="#ffaa00", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(0, y1 - 4), f"#{i}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="#ffaa00", edgecolor="none", alpha=0.9, pad=2),
        )

    try:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    except Exception as exc:
        print(f"  [warn] tight_layout failed ({exc!r}); saving without it")
        try:
            fig.savefig(out_path, dpi=150)
        except Exception as exc2:
            print(f"  [warn] savefig also failed: {exc2!r}; skipping OCR viz")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-image processing: two-stage
# --------------------------------------------------------------------------- #
def process_one_image(
    image_path: Path,
    args,
    dtd_model,
    dtd_model_name,
    dtd_needs_dct,
    device,
    processor,
    qwen_model,
    out_dir: Path,
) -> None:
    import gc
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.parse_math"] = False 

    t_start = time.time()
    stem = image_path.stem
    doc_dir = out_dir / stem

    # Defence-in-depth skip — also catches the case where process_one_image is
    # called directly without going through gather_images. Cheap and safe.
    if getattr(args, "skip_existing", False) and (doc_dir / "report.md").exists():
        print(f"\n{'='*60}\n{image_path.name}\n{'='*60}")
        print(f"  [skip] report.md already exists, nothing to do")
        return

    doc_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n{image_path.name}\n{'='*60}")

    # ---- DTD inference ----
    prob, image_pil = _dtd.infer_one_image(
        image_path, dtd_model, dtd_model_name, dtd_needs_dct, device,
        jpeg_quality=args.jpeg_quality,
    )

    # DTD heatmap overlay
    cmap = plt.get_cmap("jet")
    heat = (cmap(prob)[:, :, :3] * 255).astype(np.uint8)
    arr = np.asarray(image_pil.convert("RGB"))
    dtd_overlay_arr = (0.55 * arr + 0.45 * heat).clip(0, 255).astype(np.uint8)
    dtd_overlay_pil = Image.fromarray(dtd_overlay_arr)
    dtd_overlay_pil.save(doc_dir / "dtd_overlay.png")

    if args.save_dtd_probs:
        np.save(doc_dir / "dtd.prob.npy", prob.astype(np.float32))

    # ---- OCR ----
    print("  [ocr] running ...", flush=True)
    ocr_result = extract_ocr(
        image_path,
        gpu=torch.cuda.is_available(),
        detail=1,
        candidate_langs=args.langs,
    )
    draw_ocr_boxes(
        image_path, ocr_result["ocr_items"],
        doc_dir / "ocr_viz.png",
        title=f"OCR ({ocr_result['lang']}) - {ocr_result['n_items']} items",
    )

    dtd_hints_str = _format_dtd_hints(prob, threshold=args.dtd_threshold)
    ocr_json_str = _format_ocr_json(ocr_result)
    orig_w, orig_h = image_pil.size

    # Generation kwargs
    do_sample = not args.greedy
    gen_kwargs = dict(
        min_new_tokens=64,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=processor.tokenizer.pad_token_id
        or processor.tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(args.temperature, 1e-5)
        gen_kwargs["top_p"] = args.top_p
        if args.top_k and args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k

    # ====================================================================== #
    # STAGE 1: DTD region filtering (original + heatmap, two images)
    # ====================================================================== #
    print("\n  -- STAGE 1: DTD filtering --", flush=True)
    stage1_prompt_path = _TOOLKIT_ROOT / "prompts" / "only_dtd.txt"
    stage1_prompt = stage1_prompt_path.read_text(encoding="utf-8")

    # Append DTD bounding boxes and OCR to the prompt
    stage1_prompt = (
        stage1_prompt
        .replace("{{OCR_JSON}}", ocr_json_str)
    )
    stage1_prompt += f"\n\nDTD DETECTOR OUTPUT:\n{dtd_hints_str}"
    stage1_prompt += (
        f"\n\nIMAGE METADATA: Width={orig_w} Height={orig_h}. "
        f"All coordinates are absolute integer pixels in this coordinate system."
    )

    stage1_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "image", "image": dtd_overlay_pil},
            {"type": "text", "text": stage1_prompt},
        ],
    }]

    t1 = time.time()
    stage1_raw = _qwen_generate(qwen_model, processor, stage1_messages, gen_kwargs)
    stage1_elapsed = time.time() - t1
    print(f"\n  [stage1] generated in {stage1_elapsed:.1f}s", flush=True)

    (doc_dir / "stage1_raw.txt").write_text(stage1_raw, encoding="utf-8")
    filtered_anomalies = _extract_out(stage1_raw)
    if filtered_anomalies:
        (doc_dir / "stage1_filtered.txt").write_text(filtered_anomalies, encoding="utf-8")
        n_regions = filtered_anomalies.count("ANOMALY")
        print(f"  [stage1] filtered anomalies: {n_regions} regions")
        _draw_filtered_boxes(
            image_pil, filtered_anomalies, doc_dir / "stage1_filtered_viz.png",
            title=f"Stage 1 filtered ({n_regions} regions) - {image_path.name}",
        )
    else:
        print("  [stage1] WARNING: no <out> block found; using raw DTD hints as fallback")
        filtered_anomalies = dtd_hints_str

    # ====================================================================== #
    # STAGE 2: Final report (original only)
    # ====================================================================== #
    print("\n  -- STAGE 2: Report generation --", flush=True)
    stage2_prompt_path = _TOOLKIT_ROOT / "prompts" / "only_report.txt"
    stage2_prompt = stage2_prompt_path.read_text(encoding="utf-8")
    stage2_prompt = (
        stage2_prompt
        .replace("{{OCR_JSON}}", ocr_json_str)
        .replace("{{DTD_HINTS}}", filtered_anomalies)
    )
    stage2_prompt += (
        f"\n\nIMAGE METADATA:\n"
        f"- Width: {orig_w} pixels\n"
        f"- Height: {orig_h} pixels\n"
        f"- Coordinate system: top-left origin, x grows right, y grows down.\n"
        f"- All [GROUNDING] values MUST be absolute integer pixel coordinates "
        f"in this {orig_w}x{orig_h} image."
    )

    stage2_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": stage2_prompt},
        ],
    }]

    t2 = time.time()
    stage2_raw = _qwen_generate(qwen_model, processor, stage2_messages, gen_kwargs)
    stage2_elapsed = time.time() - t2
    print(f"\n  [stage2] generated in {stage2_elapsed:.1f}s", flush=True)

    (doc_dir / "stage2_raw.txt").write_text(stage2_raw, encoding="utf-8")
    answer = _extract_clean_report(stage2_raw)

    if not _has_report_anchor(answer):
        print("  [WARNING] Missing '# FORGERY ANALYSIS REPORT'. Using stub.")
        answer = _STUB_REPORT

    # ---- Save outputs ----
    report = parse_report(answer)
    (doc_dir / "report.md").write_text(
        answer + ("\n" if not answer.endswith("\n") else ""), encoding="utf-8",
    )
    (doc_dir / "report.json").write_text(
        json.dumps({"image_name": image_path.name, "report": answer},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Pipeline viz (simple bbox overlay)
    viz_title = (
        f"{image_path.name}   |   {report.conclusion}   |   "
        f"score={report.risk_score}   |   anomalies={len(report.anomalies)}"
    )
    _draw_boxes(image_pil, report, doc_dir / "report_viz.png", title=viz_title)

    # Full pred viz (bboxes + REASON side panels)
    try:
        visualize_report(
            image_path, doc_dir / "report.md", doc_dir / "pred_viz.png",
            title=f"Pipeline: {image_path.name}",
        )
    except Exception as exc:
        print(f"  [warn] pred viz failed: {exc!r}")

    # GT viz if available
    if args.gt_reports:
        gt_base = Path(args.gt_reports).expanduser().resolve()
        gt_path = gt_base if gt_base.is_file() else gt_base / f"{stem}_report.md"
        if gt_path.exists():
            try:
                visualize_report(
                    image_path, gt_path, doc_dir / "gt_viz.png",
                    title=f"Ground Truth: {image_path.name}",
                )
            except Exception as exc:
                print(f"  [warn] GT viz failed: {exc!r}")

    total_elapsed = time.time() - t_start
    print(
        f"\n  [done] verdict={report.conclusion}  score={report.risk_score}  "
        f"anomalies={len(report.anomalies)}  "
        f"stage1={stage1_elapsed:.1f}s  stage2={stage2_elapsed:.1f}s  "
        f"total={total_elapsed:.1f}s"
    )
    for a in report.anomalies:
        print(f"    #{a.index:03d}  {(a.type or '?'):<30}  grounding={a.grounding}")

    # Cleanup
    del stage1_messages, stage2_messages
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #
def _draw_boxes(
    image: Image.Image,
    report: ForgeryReport,
    out_path: Path,
    title: str = "",
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
            print(f"  [warn] savefig also failed: {exc2!r}; skipping OCR viz")
    plt.close(fig)

def extract_ocr(
    image_path,
    *,
    gpu: bool = True,
    detail: int = 1,
    candidate_langs: list[str] | str = ("en", "ch", "th", "ms", "id", "ar"),
    mag_ratio: float = 1.0,
) -> dict:
    if isinstance(candidate_langs, str):
        candidate_langs = [s.strip() for s in candidate_langs.split(",") if s.strip()]
    result = run_paddle_ocr_with_lang_detect(
        image_path, candidate_langs=candidate_langs, gpu=gpu,
        mag_ratio=mag_ratio, verbose=False,
    )
    result["selected_language"] = result["lang"]
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Path to a single image.")
    g.add_argument("--image_dir", help="Directory of images to process.")

    # DTD
    ap.add_argument("--config", required=True, help="DTD YAML config.")
    ap.add_argument("--checkpoint", required=True, help="DTD checkpoint .pth.")
    ap.add_argument("--dtd_threshold", type=float, default=0.4,
                    help="DTD probability threshold.")

    # OCR
    ap.add_argument("--langs", default="en,ch,th,ms,id,ar",
                    help="Comma-separated PaddleOCR language codes.")

    # Qwen
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--model_class", default="Qwen3VLForConditionalGeneration")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--attn_impl", default="sdpa",
                    choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true")

    # Output
    ap.add_argument("--out_dir", default="predictions/two_stage")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=0,
                    help="Max images from --image_dir (0=all).")
    ap.add_argument("--save_dtd_probs", action="store_true")
    ap.add_argument(
        "--skip_existing", action="store_true",
        help="Skip an image if {out_dir}/{stem}/report.md already exists. "
             "Use to resume a partially-completed batch without redoing the "
             "expensive DTD + OCR + Qwen calls.",
    )
    ap.add_argument("--gt_reports", default=None,
                    help="Path to GT report or directory of GT reports.")
    return ap.parse_args()


def gather_images(args) -> list[Path]:
    if args.image:
        return [Path(args.image).expanduser().resolve()]
    d = Path(args.image_dir).expanduser().resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)

    # Filter out already-completed images BEFORE applying --limit, so that
    # --limit N effectively means "process N more images" on resume.
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
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoProcessor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = gather_images(args)
    if not image_paths:
        print("[error] no images found")
        return 1
    print(f"[run] {len(image_paths)} image(s)")

    # Load DTD
    print("[dtd] loading model ...")
    t0 = time.time()
    _dtd._setup_paths_and_registry()
    dtd_model, dtd_model_name, dtd_needs_dct = _dtd.build_model_and_load(
        args.config, args.checkpoint, device,
    )
    print(f"[dtd] loaded {dtd_model_name} (needs_dct={dtd_needs_dct}) "
          f"in {time.time()-t0:.1f}s")

    # Load Qwen
    print("[qwen] loading processor + model ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ModelCls = getattr(transformers, args.model_class)
    dtype_map = {
        "bfloat16": torch.bfloat16, "float16": torch.float16,
        "float32": torch.float32,
    }
    qwen_model = ModelCls.from_pretrained(
        args.model_id, dtype=dtype_map[args.dtype],
        device_map=args.device_map, attn_implementation=args.attn_impl,
        trust_remote_code=True,
    ).eval()
    print(f"[qwen] loaded in {time.time()-t0:.1f}s")

    # Process
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] {image_path.name}")
        process_one_image(
            image_path=image_path, args=args,
            dtd_model=dtd_model, dtd_model_name=dtd_model_name,
            dtd_needs_dct=dtd_needs_dct, device=device,
            processor=processor, qwen_model=qwen_model,
            out_dir=out_dir,
        )

    print(f"\n[done] outputs in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())