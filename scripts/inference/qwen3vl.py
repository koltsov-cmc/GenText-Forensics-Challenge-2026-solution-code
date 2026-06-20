#!/usr/bin/env python
"""Run Qwen3-VL on a document image and produce a forgery analysis report.

Outputs: Markdown report (.md), JSON, visualization (.viz.png), thinking trace,
and raw model output. Supports various Qwen-VL model variants.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from transformers import LogitsProcessor
from tqdm.auto import tqdm
from transformers import TextStreamer


import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

# Make realtext_v2 importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from realtext_v2.report import parse_report


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Thinking"


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
PROMPT = """You are a document-forgery forensic analyst. Examine the attached \
image and produce a FORGERY ANALYSIS REPORT.

Return ONLY a Markdown report following EXACTLY this schema - no preamble, \
no code fences, no commentary:

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
- Bias toward DETECTION, not toward dismissing evidence. Document forgery is
  often subtle. If you see ANY concrete visual or textual anomaly, treat it
  as a candidate forgery and emit an ANOMALY block. Do NOT explain anomalies
  away as "common design", "standard practice", or "blurred for privacy"
  unless the evidence is unambiguous. False negatives are far costlier than
  false positives in this task.
- Every ANOMALY must be grounded in a concrete observation -- not vibes,
  not speculation.
- Textual evidence is just as valid as visual evidence. A semantic
  contradiction, logical inconsistancy, an impossible value, or a domain-inconsistent term
  is a legitimate ANOMALY even if there is no visual artifact at the
  same location. Use [GROUNDING] to point to the bounding box of the
  offending text.


Look for ALL of the following manipulation types:

  VISUAL AND TYPOGRAPHIC CUES:
    - Font mismatch (typeface, weight, italic, OR slight rendering
      differences in stroke width, anti-aliasing, kerning, baseline).
    - Inconsistent character spacing or sudden indentation changes
      within a word, line, or block of otherwise-uniform text.
    - Misalignment with the surrounding baseline or margin grid.
    - Local blur, sharpness, or compression artifacts that differ from
      the rest of the page (JPEG block boundaries, halo rings, smudges). For logos and text in document.
    - Pixel-level seams, double-edges, or mismatched anti-aliasing where
      content was pasted in.
    - Solid-color rectangles, possibly covering text
      or a field -- these are likely redactions.
    - Painted-over or smeared regions, color/brightness patches that
      don't match the paper background.
    - Tilt, rotation, or warping of a small region relative to the
      surrounding content.
    - Copy-move duplication: identical glyph shapes or stamps appearing
      in multiple positions where they should differ.
    - And other

  CONTENT AND LOGICAL CUES:
    - Numerical inconsistencies (totals that don't add up, dates that
      contradict each other, mismatched amounts in figures vs. words, and other).
    - Clearly impossible or absurd values (5:00 a.m. business meeting,
      birthdate in the future, ZIP code with wrong digit count).
    - Internal contradictions between fields of the same document
      (name in one field doesn't match the same name elsewhere).
    - Mixed languages or scripts in places where one would expect
      uniformity, when accompanied by visual cues above.
    - Semantic contradictions or oxymora: terms inside the document that
    contradict each other.
    - Implausible role / authority combinations (e.g. "Junior CEO",
    "Acting Permanent Director"), or fields where the value type
    is wrong for the field (a date in a name field, a person's name
    in an amount field).
    - Domain-impossible values: a medical form listing a non-existent
    drug, a legal form citing a non-existent statute, a tax form
    using fields from a different country's tax system. If you
    recognize the domain, apply common-sense domain knowledge.
    - AND OTHER.

If you find no concrete tampering evidence after a thorough check, output
Conclusion AUTHENTIC with a low RISK_SCORE (0-15) and explain in the SUMMARY
which categories you checked and ruled out.

It is implied that in an ideal document everything should be good from a visual and semantic point of view.

Analysis procedure:

Before producing the report, perform a SYSTEMATIC TOP-TO-BOTTOM PASS over
the document. Do this in your <think> reasoning.

At the very beginning, read the document completely and understand its essence, 
its semantic content, and the domain it belongs to. Also, get a rough idea of ​​the document's overall style.

Next, read the entire document, top to bottom, left to right, WORD BY WORD. 
Check every word for VISUAL AND TYPOGRAPHIC CUES.

Then, after reading a paragraph or several sentences, go over them or the entire paragraph 
again and check for CONTENT AND LOGICAL CUES in the context of that paragraph, the ENTIRE DOCUMENT, and YOUR OWN KNOWLEDGE.

As you check, highlight and memorize any possible ANOMALIES.

When you have checked all the words VISUAL AND TYPOGRAPHIC CUES and all the paragraphs for CONTENT AND LOGICAL CUES, 
go through the possible ANOMALIES you have found one last time and make sure that everything is ok, 
and then write a report in the required format.
"""


# --------------------------------------------------------------------------- #
# Qwen smart-resize  (keeps coords the model emits aligned with our image)
# --------------------------------------------------------------------------- #
def smart_resize(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = 28,
) -> tuple[int, int]:
    """Canonical Qwen smart_resize: snap to multiples of ``factor`` while
    keeping the total pixel count within [min_pixels, max_pixels] and the
    aspect ratio as close as possible to the original."""
    if height < factor or width < factor:
        # Too small -- blow up to at least `factor` in each dim.
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
# Thinking extraction
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_REPORT_ANCHOR_RE = re.compile(
    r"#\s*FORGERY\s+ANALYSIS\s+REPORT", re.IGNORECASE
)


def strip_thinking(text: str) -> tuple[str, str]:
    """Return (thinking_trace, clean_answer).

    Strategy, in order:
      1. If `<think>...</think>` tags exist, peel them out.
      2. Otherwise, find the first '# FORGERY ANALYSIS REPORT' marker and
         treat everything before it as stray reasoning.
      3. If the marker is missing entirely, keep the whole output as the
         answer so the caller can at least inspect it (but callers should
         warn the user).
    """
    thinking_parts = _THINK_RE.findall(text)
    answer = _THINK_RE.sub("", text).strip()

    if not thinking_parts:
        m = _REPORT_ANCHOR_RE.search(answer)
        if m and m.start() > 0:
            thinking_parts = [answer[: m.start()].strip()]
            answer = answer[m.start():].strip()

    # Strip optional ```markdown fences.
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
# Visualisation
# --------------------------------------------------------------------------- #
def draw_boxes(
    image: Image.Image,
    report,
    out_path: Path,
    title: str = "",
) -> None:
    w, h = image.size
    # Sensible figure size: keep aspect, cap total width.
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
        # Defensive: swap if model emitted them in a weird order.
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
            bbox=dict(
                facecolor="#ff2e2e",
                edgecolor="none",
                alpha=0.9,
                pad=2,
            ),
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image", required=True, help="Path to the image file.")
    ap.add_argument("--out_dir", default="predictions",
                    help="Directory for all outputs.")
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument(
        "--model_class", default="Qwen3VLForConditionalGeneration",
        help="transformers class name (swap to Qwen2_5_VLForConditionalGeneration for Qwen2.5).",
    )
    ap.add_argument("--max_new_tokens", type=int, default=10384,
                    help="Thinking models need a lot. 8192 may truncate; 16384 is safer.")
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--attn_impl", default="sdpa",
                    choices=["eager", "sdpa", "flash_attention_2"])
    # Sampling: official Qwen3-VL-Thinking recipe (temp=1.0, top_p=0.95, top_k=20).
    # DO NOT use greedy (temperature=0) for Thinking variants -- it breaks CoT.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true",
                    help="Override sampling and use greedy decoding "
                         "(NOT recommended for Thinking variants).")
    # WARNING: repetition penalties break Thinking-mode reports because the
    # schema itself has repeated n-grams ('# FORGERY ANALYSIS REPORT',
    # '[GROUNDING]', etc.). Default OFF.
    ap.add_argument("--repetition_penalty", type=float, default=1.0,
                    help="Leave at 1.0 unless you see actual loops. "
                         "Values >1.0 may corrupt the schema in Thinking models.")
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0,
                    help="Leave at 0 for Thinking models. The schema has "
                         "repeated n-grams which n-gram blocking will corrupt.")
    ap.add_argument("--prompt", default=None, help="Override the default prompt.")
    ap.add_argument("--no_resize", action="store_true",
                    help="Skip smart-resize. Coords may not align with the image.")
    return ap.parse_args()


class TqdmLogitsProcessor(LogitsProcessor):
    def __init__(self, total: int, desc: str = "generating"):
        self.pbar = tqdm(total=total, desc=desc, unit="tok")
    def __call__(self, input_ids, scores):
        self.pbar.update(1)
        return scores
    def close(self):
        self.pbar.close()

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # Load image.
    image = Image.open(str(image_path)).convert("RGB")
    orig_w, orig_h = image.size

    if args.no_resize:
        model_image = image
    else:
        new_h, new_w = smart_resize(
            orig_h, orig_w,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        if (new_w, new_h) != (orig_w, orig_h):
            print(f"smart_resize: {orig_w}x{orig_h} -> {new_w}x{new_h}")
            model_image = image.resize((new_w, new_h), Image.BILINEAR)
        else:
            model_image = image

    # Heavy imports (after arg parsing so --help is instant).
    import torch
    import transformers
    from transformers import AutoProcessor

    ModelCls = getattr(transformers, args.model_class)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    print(f"loading model:     {args.model_id}  (dtype={args.dtype})")
    model = ModelCls.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )
    model.eval()

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
    prompt = (args.prompt or PROMPT) + size_hint

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": model_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Remove keys unused by the VL forward pass (avoids warnings on some versions).
    inputs.pop("token_type_ids", None)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}

    print(f"generating (max_new_tokens={args.max_new_tokens})...")
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
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    with torch.inference_mode():
        streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        #tqdm_proc = TqdmLogitsProcessor(total=args.max_new_tokens, desc="qwen3-vl")
        #gen_kwargs["logits_processor"] = [tqdm_proc]

        out_ids = model.generate(**inputs, **gen_kwargs, streamer=streamer)
        #tqdm_proc.close()


    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
    texts = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    raw_text = texts[0]
    thinking, answer = strip_thinking(raw_text)

    schema_ok = has_valid_report_anchor(answer)
    if not schema_ok:
        print(
            "\n[WARNING] Model output does not contain '# FORGERY ANALYSIS REPORT'."
            "\n          Writing stub report; inspect .raw.txt for the full output."
            "\n          Usually this means the model got stuck in Thinking."
            "\n          Retry with --model_id Qwen/Qwen3-VL-2B-Instruct"
            " or increase --max_new_tokens.",
        )
        answer = STUB_REPORT

    # Parse + save artefacts.
    report = parse_report(answer)

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(answer + ("\n" if not answer.endswith("\n") else ""),
                       encoding="utf-8")

    json_obj = {"image_name": image_path.name, "report": answer}
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(json_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw_path = out_dir / f"{stem}.raw.txt"
    raw_path.write_text(raw_text, encoding="utf-8")

    thinking_path = None
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

    # Console summary.
    bar = "=" * 72
    print("\n" + bar)
    print(f"IMAGE:      {image_path}")
    print(f"VERDICT:    {report.conclusion}")
    print(f"RISK_SCORE: {report.risk_score}")
    print(f"ANOMALIES:  {len(report.anomalies)}")
    for a in report.anomalies:
        print(f"   #{a.index:03d}  {(a.type or '?'):<30}  grounding={a.grounding}")
    print(bar)
    print(json.dumps(json_obj, ensure_ascii=False, indent=2))
    print(bar)
    print(f"md      -> {md_path}")
    print(f"json    -> {json_path}")
    print(f"viz     -> {viz_path}")
    print(f"raw     -> {raw_path}")
    if thinking_path:
        print(f"think   -> {thinking_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())