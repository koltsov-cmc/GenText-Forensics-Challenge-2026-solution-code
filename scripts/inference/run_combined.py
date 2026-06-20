#!/usr/bin/env python
"""Combined DTD + Qwen3-VL pipeline for document forgery detection.

Runs DTD on the image for tampering probability, then feeds results to Qwen3-VL
for a structured forensic report. Two fusion modes: "coordinates" (text hints)
and "image" (heatmap overlay as second image).
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

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Auto-detect repo root (script may live in scripts/ or repo root)
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = Path(__file__).resolve().parent
_possible_roots = [_SCRIPT_DIR, _SCRIPT_DIR.parent]
_TOOLKIT_ROOT = None
for r in _possible_roots:
    if (r / "realtext_v2").is_dir() or (r / "models").is_dir():
        _TOOLKIT_ROOT = r
        break
if _TOOLKIT_ROOT is None:
    _TOOLKIT_ROOT = _SCRIPT_DIR.parent  # sane fallback

sys.path.insert(0, str(_TOOLKIT_ROOT))

from realtext_v2.grounding import mask_to_boxes
from realtext_v2.report import parse_report, ForgeryReport, Anomaly

# --------------------------------------------------------------------------- #
# Re-use DTD inference machinery
# --------------------------------------------------------------------------- #
_DTD_SCRIPT_DIR = _TOOLKIT_ROOT / "realtext_v2_toolkit" /"ForensicHub" / "dtd_train"
sys.path.insert(0, str(_DTD_SCRIPT_DIR))

print("_DTD_SCRIPT_DIR", _DTD_SCRIPT_DIR)
print("_TOOLKIT_ROOT", _TOOLKIT_ROOT)

import run_doc_forensics_inference as _dtd  # noqa: E402

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_BASE_PROMPT = """You are a document-forgery forensic analyst. Examine the attached image(s) and produce a FORGERY ANALYSIS REPORT.

Return ONLY a Markdown report following EXACTLY this schema - no preamble, no code fences, no commentary:

# FORGERY ANALYSIS REPORT

**[Conclusion]:** <FORGED or AUTHENTIC>
**[RISK_SCORE]:** <integer 0-100>

### ANOMALY_001: <short type> (<short location>)
[GROUNDING]: [xmin, ymin, xmax, ymax]
[REASON]: <concise natural-language justification grounded in visible evidence>

### ANOMALY_002: <short type> (<short location>)
[GROUNDING]: [xmin, ymin, xmax, ymax]
[REASON]: ...

## SUMMARY
<one short paragraph summarising the verdict and key evidence>

Detection policy:
- Bias toward DETECTION, not toward dismissing evidence. Document forgery is often subtle. If you see ANY concrete visual or textual anomaly, treat it as a candidate forgery and emit an ANOMALY block. Do NOT explain anomalies away as "common design", "standard practice", or "blurred for privacy" unless the evidence is unambiguous. False negatives are far costlier than false positives.
- Every ANOMALY must be grounded in a concrete observation -- not vibes, not speculation.
- Textual evidence is just as valid as visual evidence. A semantic contradiction, logical inconsistency, an impossible value, or a domain-inconsistent term is a legitimate ANOMALY even if there is no visual artifact at the same location. Use [GROUNDING] to point to the bounding box of the offending text.

Look for ALL of the following manipulation types:

  VISUAL AND TYPOGRAPHIC CUES:
    - Font mismatch (typeface, weight, italic, OR slight rendering differences in stroke width, anti-aliasing, kerning, baseline).
    - Inconsistent character spacing or sudden indentation changes within a word, line, or block of otherwise-uniform text.
    - Misalignment with the surrounding baseline or margin grid.
    - Local blur, sharpness, or compression artifacts that differ from the rest of the page (JPEG block boundaries, halo rings, smudges). For logos and text in document.
    - Pixel-level seams, double-edges, or mismatched anti-aliasing where content was pasted in.
    - Solid-color rectangles, possibly covering text or a field -- these are likely redactions.
    - Painted-over or smeared regions, color/brightness patches that don't match the paper background.
    - Tilt, rotation, or warping of a small region relative to the surrounding content.
    - Copy-move duplication: identical glyph shapes or stamps appearing in multiple positions where they should differ.

  CONTENT AND LOGICAL CUES:
    - Numerical inconsistencies (totals that don't add up, dates that contradict each other, mismatched amounts in figures vs. words).
    - Clearly impossible or absurd values (5:00 a.m. business meeting, birthdate in the future, ZIP code with wrong digit count).
    - Internal contradictions between fields of the same document (name in one field doesn't match the same name elsewhere).
    - Mixed languages or scripts in places where one would expect uniformity, when accompanied by visual cues above.
    - Semantic contradictions or oxymora: terms inside the document that contradict each other.
    - Implausible role / authority combinations (e.g. "Junior CEO", "Acting Permanent Director"), or fields where the value type is wrong for the field.
    - Domain-impossible values: a medical form listing a non-existent drug, a legal form citing a non-existent statute, a tax form using fields from a different country's tax system.

If you find no concrete tampering evidence after a thorough check, output Conclusion AUTHENTIC with a low RISK_SCORE (0-15) and explain in the SUMMARY which categories you checked and ruled out.

It is implied that in an ideal document everything should be good from a visual and semantic point of view.

Analysis procedure:
Before producing the report, perform a SYSTEMATIC TOP-TO-BOTTOM PASS over the document. Do this in your <think> reasoning.
At the very beginning, read the document completely and understand its essence, its semantic content, and the domain it belongs to. Also, get a rough idea of the document's overall style.
Next, read the entire document, top to bottom, left to right, WORD BY WORD. Check every word for VISUAL AND TYPOGRAPHIC CUES.
Then, after reading a paragraph or several sentences, go over them or the entire paragraph again and check for CONTENT AND LOGICAL CUES in the context of that paragraph, the ENTIRE DOCUMENT, and YOUR OWN KNOWLEDGE.
As you check, highlight and memorize any possible ANOMALIES.
When you have checked all the words for VISUAL AND TYPOGRAPHIC CUES and all the paragraphs for CONTENT AND LOGICAL CUES, go through the possible ANOMALIES you have found one last time and make sure that everything is ok, and then write a report in the required format.
"""

_COORDINATES_PREFIX = (
    "An auxiliary pixel-level detector (DTD) has pre-scanned the document. "
    "The following regions were flagged as suspicious (threshold 0.4). "
    "Use them as HINTS only — verify visually, as the detector may produce false positives.\n\n"
)

_IMAGE_PREFIX = (
    "Image 1 is the document under analysis. "
    "Image 2 is a tampering-probability heatmap from an auxiliary detector (DTD). "
    "Red / yellow regions indicate higher suspicion of tampering. "
    "Use the heatmap as a HINT, but verify visually — the detector may produce false positives.\n\n"
)


# --------------------------------------------------------------------------- #
# Qwen smart-resize (copied from run_qwen3vl_inference.py)
# --------------------------------------------------------------------------- #
def smart_resize(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = 28,
) -> tuple[int, int]:
    if height < factor or width < factor:
        height = max(height, factor)
        width = max(width, factor)
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# --------------------------------------------------------------------------- #
# Thinking extraction (copied from run_qwen3vl_inference.py)
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_REPORT_ANCHOR_RE = re.compile(
    r"#\s*FORGERY\s+ANALYSIS\s+REPORT", re.IGNORECASE
)


def strip_thinking(text: str) -> tuple[str, str]:
    thinking_parts = _THINK_RE.findall(text)
    answer = _THINK_RE.sub("", text).strip()
    if not thinking_parts:
        m = _REPORT_ANCHOR_RE.search(answer)
        if m and m.start() > 0:
            thinking_parts = [answer[: m.start()].strip()]
            answer = answer[m.start():].strip()
    answer = re.sub(r"^```(?:markdown|md)?\s*", "", answer)
    answer = re.sub(r"\s*```$", "", answer)
    thinking = "\n\n".join(p.strip() for p in thinking_parts if p.strip()).strip()
    return thinking, answer.strip()


def has_valid_report_anchor(text: str) -> bool:
    return bool(_REPORT_ANCHOR_RE.search(text))


STUB_REPORT = """# FORGERY ANALYSIS REPORT

**[Conclusion]:** AUTHENTIC
**[RISK_SCORE]:** 0

## SUMMARY
Model failed to produce a schema-compliant report (see .raw.txt).
"""


# --------------------------------------------------------------------------- #
# Visualisation (copied from run_qwen3vl_inference.py)
# --------------------------------------------------------------------------- #
def draw_boxes(
    image: Image.Image,
    report: ForgeryReport,
    out_path: Path,
    title: str = "",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

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
            (x1, y1),
            max(1, x2 - x1),
            max(1, y2 - y1),
            linewidth=2,
            edgecolor="#ff2e2e",
            facecolor="none",
        )
        ax.add_patch(rect)
        label_bits = [f"#{a.index}"]
        if a.type:
            label_bits.append(a.type[:28])
        ax.text(
            x1,
            max(0, y1 - 6),
            "  ".join(label_bits),
            color="white",
            fontsize=9,
            bbox=dict(facecolor="#ff2e2e", edgecolor="none", alpha=0.9, pad=2),
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# DTD helpers
# --------------------------------------------------------------------------- #
def run_dtd_inference(
    image_path: Path,
    model,
    model_name: str,
    needs_dct: bool,
    device,
    out_dir: Path,
    stem: str,
    jpeg_quality: int = 95,
) -> tuple[np.ndarray, Image.Image, Path]:
    """Run DTD on a single image and save intermediate artefacts.

    Args:
        model: already-loaded DTD model.
        model_name: name string (e.g. "DTD").
        needs_dct: whether DCT input is required.

    Returns:
        prob_map: float32 [H, W]
        image_pil: the (possibly resized) PIL image used by DTD
        overlay_path: Path to the saved overlay PNG
    """
    print(f"    [dtd] calling infer_one_image ...", flush=True)
    prob, image_pil = _dtd.infer_one_image(
        image_path, model, model_name, needs_dct, device,
        jpeg_quality=jpeg_quality,
    )
    print(f"    [dtd] infer_one_image returned", flush=True)

    # Save intermediate outputs for debugging / reuse
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}.dtd.prob.npy", prob.astype(np.float32))
    print(f"    [dtd-save] prob.npy saved", flush=True)

    # Build heatmap and overlay using matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("jet")
    heat = (cmap(prob)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(heat).save(out_dir / f"{stem}.dtd.heatmap.png")
    print(f"    [dtd-save] heatmap.png saved", flush=True)

    arr = np.asarray(image_pil.convert("RGB"))
    overlay = (0.55 * arr + 0.45 * heat).clip(0, 255).astype(np.uint8)
    overlay_path = out_dir / f"{stem}.dtd.overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    print(f"    [dtd-save] overlay.png saved", flush=True)

    # Binary mask at threshold 0.4 (used for coordinate extraction)
    mask_04 = ((prob >= 0.4) * 255).astype(np.uint8)
    Image.fromarray(mask_04, mode="L").save(out_dir / f"{stem}.dtd.mask_t40.png")
    print(f"    [dtd-save] mask_t40.png saved", flush=True)

    return prob, image_pil, overlay_path


def build_coordinate_hint(prob: np.ndarray, threshold: float = 0.4) -> str:
    """Extract connected-component boxes from the probability map and
    format them as a text hint for the VLM prompt."""
    mask = (prob >= threshold).astype(np.uint8) * 255
    boxes = mask_to_boxes(mask, min_area=200)
    if not boxes:
        return "The auxiliary DTD detector found NO suspicious regions above the threshold.\n\n"

    lines = [f"DTD flagged {len(boxes)} suspicious region(s):"]
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        sub = prob[y1:y2, x1:x2]
        conf = float(sub.mean()) if sub.size else 0.0
        lines.append(
            f"  Region {i}: [{x1}, {y1}, {x2}, {y2}] "
            f"(mean confidence {conf:.3f})"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Qwen3-VL helpers
# --------------------------------------------------------------------------- #
def build_messages_for_qwen(
    fuse: str,
    model_image: Image.Image,
    overlay_image: Optional[Image.Image],
    base_prompt: str,
    size_hint: str,
    coordinate_hint: str,
) -> list[dict]:
    """Build the message list for Qwen3-VL depending on fusion mode."""
    if fuse == "image":
        # Two images: document + heatmap overlay
        content = [
            {"type": "image", "image": model_image},
            {"type": "image", "image": overlay_image},
            {"type": "text", "text": _IMAGE_PREFIX + base_prompt + size_hint},
        ]
    else:
        # Single image + coordinate text hint
        content = [
            {"type": "image", "image": model_image},
            {"type": "text", "text": _COORDINATES_PREFIX + coordinate_hint + base_prompt + size_hint},
        ]
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Path to a single image.")
    g.add_argument("--image_dir", help="Directory of images to process.")

    # DTD
    ap.add_argument("--config", required=True,
                    help="YAML training config for DTD (model section).")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to trained DTD .pth checkpoint.")

    # Fusion mode
    ap.add_argument("--fuse", required=True, choices=["coordinates", "image"],
                    help="How to fuse DTD evidence into Qwen.")

    # Qwen
    ap.add_argument(
        "--qwen_model_id",
        default=str(_TOOLKIT_ROOT / "models" / "Qwen3-VL-32B-Thinking"),
        help="HF id or local path of the Qwen3-VL model. "
             "Defaults to the local copy in models/Qwen3-VL-32B-Thinking.",
    )
    ap.add_argument("--qwen_model_class", default="Qwen3VLForConditionalGeneration")
    ap.add_argument("--qwen_dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--qwen_device_map", default="auto")
    ap.add_argument("--qwen_attn_impl", default="sdpa",
                    choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--max_new_tokens", type=int, default=16384,
                    help="Thinking models need a lot of tokens.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true",
                    help="Use greedy decoding (NOT recommended for Thinking).")

    # Image sizing
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--no_resize", action="store_true",
                    help="Skip smart-resize for the document image.")

    # Output
    ap.add_argument("--out_dir", default="predictions/combined")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=0,
                    help="Max images to process from --image_dir (0=all).")
    ap.add_argument("--dtd_threshold", type=float, default=0.4,
                    help="Probability threshold for coordinate extraction.")
    ap.add_argument("--keep_dtd_artefacts", action="store_true", default=True,
                    help="Save prob.npy, overlay.png, mask.png from DTD.")
    return ap.parse_args()


def gather_images(args) -> list[Path]:
    if args.image:
        return [Path(args.image).expanduser().resolve()]
    d = Path(args.image_dir).expanduser().resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
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
    print(f"[run] {len(image_paths)} image(s)  fuse={args.fuse}")

    # ------------------------------------------------------------------ #
    # Load DTD model once
    # ------------------------------------------------------------------ #
    print("[dtd] loading model ...")
    t0 = time.time()
    _dtd._setup_paths_and_registry()
    dtd_model, dtd_model_name, dtd_needs_dct = _dtd.build_model_and_load(
        args.config, args.checkpoint, device
    )
    print(f"[dtd] loaded {dtd_model_name} (needs_dct={dtd_needs_dct}) "
          f"in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------ #
    # Load Qwen model once
    # ------------------------------------------------------------------ #
    print("[qwen] loading processor + model ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        args.qwen_model_id, trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ModelCls = getattr(transformers, args.qwen_model_class)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    qwen_model = ModelCls.from_pretrained(
        args.qwen_model_id,
        dtype=dtype_map[args.qwen_dtype],
        device_map=args.qwen_device_map,
        attn_implementation=args.qwen_attn_impl,
        trust_remote_code=True,
    ).eval()
    print(f"[qwen] loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------ #
    # Process each image
    # ------------------------------------------------------------------ #
    for idx, image_path in enumerate(image_paths, start=1):
        stem = image_path.stem
        print(f"\n[{idx}/{len(image_paths)}] {image_path.name}")

        # ---- DTD inference ----
        dtd_out_dir = out_dir / "dtd_cache" if args.keep_dtd_artefacts else out_dir
        dtd_out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        print(f"  [dtd] starting inference ...", flush=True)
        prob, image_pil, overlay_path = run_dtd_inference(
            image_path, dtd_model, dtd_model_name, dtd_needs_dct, device,
            dtd_out_dir, stem, jpeg_quality=args.jpeg_quality,
        )
        print(f"  [dtd] done in {time.time()-t0:.1f}s  prob shape={prob.shape}  "
              f"max={float(prob.max()):.4f}  mean={float(prob.mean()):.4f}", flush=True)

        # ---- Prepare image for Qwen ----
        orig_w, orig_h = image_pil.size
        if args.no_resize:
            model_image = image_pil
        else:
            new_h, new_w = smart_resize(
                orig_h, orig_w,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
            if (new_w, new_h) != (orig_w, orig_h):
                print(f"  smart_resize: {orig_w}x{orig_h} -> {new_w}x{new_h}")
                model_image = image_pil.resize((new_w, new_h), Image.BILINEAR)
            else:
                model_image = image_pil

        img_w, img_h = model_image.size
        size_hint = (
            f"\n\nIMAGE METADATA:\n"
            f"- Width: {img_w} pixels\n"
            f"- Height: {img_h} pixels\n"
            f"- Coordinate system: top-left origin, x grows right, y grows down.\n"
            f"- All [GROUNDING] values MUST be absolute integer pixel coordinates "
            f"in this {img_w}x{img_h} image, with 0 <= xmin < xmax <= {img_w} "
            f"and 0 <= ymin < ymax <= {img_h}.\n"
            f"- Do NOT use normalized [0..1] or [0..1000] coordinates."
        )

        # ---- Build hint depending on fuse mode ----
        if args.fuse == "coordinates":
            coordinate_hint = build_coordinate_hint(prob, threshold=args.dtd_threshold)
            messages = build_messages_for_qwen(
                fuse="coordinates",
                model_image=model_image,
                overlay_image=None,
                base_prompt=_BASE_PROMPT,
                size_hint=size_hint,
                coordinate_hint=coordinate_hint,
            )
        else:
            # For image mode, ensure overlay is same size as model_image
            overlay_pil = Image.open(str(overlay_path)).convert("RGB")
            if overlay_pil.size != model_image.size:
                overlay_pil = overlay_pil.resize(model_image.size, Image.BILINEAR)
            messages = build_messages_for_qwen(
                fuse="image",
                model_image=model_image,
                overlay_image=overlay_pil,
                base_prompt=_BASE_PROMPT,
                size_hint=size_hint,
                coordinate_hint="",
            )

        # ---- Qwen generation ----
        t0 = time.time()
        print(f"  [qwen] building chat template ...", flush=True)
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
        print(f"  [qwen] input_ids shape={inputs['input_ids'].shape}  "
              f"device={inputs['input_ids'].device}", flush=True)

        do_sample = not args.greedy
        gen_kwargs: dict = dict(
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

        print(f"  [qwen] starting generate()  max_new_tokens={args.max_new_tokens}  "
              f"do_sample={do_sample} ...  (this may take a while for 32B)", flush=True)
        from transformers import TextStreamer
        streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            out_ids = qwen_model.generate(**inputs, **gen_kwargs, streamer=streamer)
        print(f"\n  [qwen] generate() finished  output_len={out_ids.shape[1]}  "
              f"new_tokens={out_ids.shape[1]-inputs['input_ids'].shape[1]}", flush=True)

        trimmed = [o[len(iid):] for iid, o in zip(inputs["input_ids"], out_ids)]
        texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw_text = texts[0]
        thinking, answer = strip_thinking(raw_text)

        schema_ok = has_valid_report_anchor(answer)
        if not schema_ok:
            print("  [WARNING] Model output missing '# FORGERY ANALYSIS REPORT'.")
            answer = STUB_REPORT

        # ---- Parse and save artefacts ----
        report = parse_report(answer)

        md_path = out_dir / f"{stem}.md"
        md_path.write_text(answer + ("\n" if not answer.endswith("\n") else ""), encoding="utf-8")

        json_obj = {"image_name": image_path.name, "report": answer}
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        raw_path = out_dir / f"{stem}.raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8")

        if thinking:
            thinking_path = out_dir / f"{stem}.thinking.txt"
            thinking_path.write_text(thinking, encoding="utf-8")

        viz_path = out_dir / f"{stem}.viz.png"
        viz_title = (
            f"{image_path.name}   |   "
            f"{report.conclusion}   |   "
            f"score={report.risk_score}   |   "
            f"anomalies={len(report.anomalies)}"
        )
        draw_boxes(model_image, report, viz_path, title=viz_title)

        print(f"  [qwen] {time.time()-t0:.1f}s  verdict={report.conclusion}  "
              f"score={report.risk_score}  anomalies={len(report.anomalies)}")
        for a in report.anomalies:
            print(f"    #{a.index:03d}  {(a.type or '?'):<30}  grounding={a.grounding}")

    print(f"\n[done] outputs in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
