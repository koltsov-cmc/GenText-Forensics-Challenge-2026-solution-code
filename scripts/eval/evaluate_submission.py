#!/usr/bin/env python
"""Comprehensive 4-metric evaluator for the GenText-Forensics challenge.

Computes SDet (detection F1), SLoc (pixel grounding IoU/F1), SExp (BERTScore),
and SRep (LLM Judge with Qwen3-VL-32B-Instruct). Supports batch mode (preds
JSONL), folder mode (per-document subfolders), and single-pair judge mode.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Toolkit imports (only needed in batch mode).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =========================================================================== #
# LLM-Judge rubric prompt
# =========================================================================== #
RUBRIC_PROMPT_TEMPLATE = """\
You are an expert forensic-document analyst acting as an impartial judge for the
GenText-Forensics challenge. You will be shown TWO Markdown forgery-analysis
reports describing the SAME image:

  - GROUND TRUTH (GT): the expert-annotated reference.
  - CANDIDATE: a system-generated report whose quality you must score.

Both reports follow this schema:
  * **[Conclusion]:** either FORGED or AUTHENTIC
  * **[RISK_SCORE]:** integer 0-100
  * Zero or more anomaly entries headed `### ANOMALY_XXX`, each containing:
      - `[GROUNDING]:` `[xmin, ymin, xmax, ymax]` (normalised, 0-1)
      - `[REASON]:`   natural-language justification
  * A final `## SUMMARY` section synthesising the findings.

Score the CANDIDATE on three rubric dimensions. Each is an INTEGER from 0 to
100. Use the full range; do not cluster everything in one band.

============================================================================
1. factuality  — accuracy of the verdict and the claimed evidence
============================================================================
  * Does the candidate's `[Conclusion]` match GT? (FORGED vs AUTHENTIC mismatch
    is a severe penalty: cap factuality at 25.)
  * Are the candidate's anomaly claims (artefacts described in `[REASON]`,
    regions in `[GROUNDING]`) consistent with what GT describes for the same
    or overlapping regions?
  * Penalise hallucinated anomalies (claims with no GT counterpart) and
    factually wrong descriptions (e.g. "blue text" when GT says black).
  * If GT is AUTHENTIC and CANDIDATE invents forgeries, factuality is very
    low (typically 0-20).
  * RISK_SCORE direction should agree with the verdict (FORGED -> high,
    AUTHENTIC -> low); inconsistency is a minor penalty.

============================================================================
2. reasoning  — logical deduction from visual clues
============================================================================
  * Does each `[REASON]` argue from CONCRETE observable evidence (typography,
    kerning, baseline, edges, halo/JPEG artefacts, lighting, colour cast,
    semantic contradiction) to the conclusion?
  * Is the argument coherent, non-circular, and free of internal contradiction?
  * Generic boilerplate ("looks suspicious", "appears tampered") with no
    specific observation should score low (<= 40), even if the verdict happens
    to be right.
  * Reward chains that connect specific cues to the specific conclusion.

============================================================================
3. completeness  — coverage and format compliance
============================================================================
  * Does the candidate identify ALL anomaly regions present in GT?
    Missing anomalies are the primary penalty here. Extra (hallucinated)
    anomalies do NOT belong in this dimension — they are a factuality issue.
  * Are all mandatory tags present and well-formed:
    `**[Conclusion]:**`, `**[RISK_SCORE]:**`, `[GROUNDING]:`, `[REASON]:`,
    a `SUMMARY` section?
  * Is the structure parseable (no broken Markdown, coordinates as
    `[x1, y1, x2, y2]`, etc.)?
  * For AUTHENTIC GT: completeness is high if the candidate also says
    AUTHENTIC, follows the schema, and provides a coherent SUMMARY (no
    anomaly entries are required).

============================================================================
Scoring bands (apply per dimension)
============================================================================
   90-100  : excellent — closely matches GT in substance and form.
   70-89   : mostly correct, minor gaps or formatting slips.
   50-69   : partially correct, several notable issues.
   30-49   : weak — major omissions, errors, or boilerplate.
    0-29   : incorrect, mostly hallucinated, or unparseable.

============================================================================
Output format (STRICT)
============================================================================
Return ONE JSON object on a single line, with EXACTLY these three integer
keys, and NOTHING ELSE — no prose, no Markdown fences, no commentary:

{{"factuality": <int 0-100>, "reasoning": <int 0-100>, "completeness": <int 0-100>}}

============================================================================
GROUND TRUTH REPORT
============================================================================
{gt_text}

============================================================================
CANDIDATE REPORT
============================================================================
{pred_text}

============================================================================
Now output the JSON object only.
"""


# =========================================================================== #
# Robust JSON parsing for judge output
# =========================================================================== #
_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)
_FIELD_RES = {
    "factuality":   re.compile(r'"?factuality"?\s*[:=]\s*(\d{1,3})', re.I),
    "reasoning":    re.compile(r'"?reasoning"?\s*[:=]\s*(\d{1,3})',  re.I),
    "completeness": re.compile(r'"?completeness"?\s*[:=]\s*(\d{1,3})', re.I),
}


def _clip_score(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return float(max(0, min(100, v)))


def parse_judge_response(text: str) -> dict[str, float]:
    """Parse judge output into {factuality, reasoning, completeness} (0-100).

    Strategy:
      1. Try direct json.loads.
      2. Try to extract the first {...} block via regex, then json.loads it.
      3. Per-field regex fallback.
      4. Return zeros if nothing usable found.
    """
    out = {"factuality": 0.0, "reasoning": 0.0, "completeness": 0.0}

    if not text:
        return out

    # 1) Direct parse.
    for candidate in (text.strip(),):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                norm = {k.lower(): v for k, v in obj.items()}
                if any(k in norm for k in out):
                    for k in out:
                        if k in norm:
                            out[k] = _clip_score(norm[k])
                    return out
        except json.JSONDecodeError:
            pass

    # 2) Extract the first {...} blob, including nested newlines.
    for m in _JSON_OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                norm = {k.lower(): v for k, v in obj.items()}
                if any(k in norm for k in out):
                    for k in out:
                        if k in norm:
                            out[k] = _clip_score(norm[k])
                    return out
        except json.JSONDecodeError:
            continue

    # 3) Per-field regex fallback.
    found_any = False
    for k, rx in _FIELD_RES.items():
        m = rx.search(text)
        if m:
            out[k] = _clip_score(m.group(1))
            found_any = True

    if not found_any:
        # Caller may want to log this; keep it cheap here.
        pass
    return out


# =========================================================================== #
# Custom rubric scoring (replaces toolkit rubric_score so we control the prompt)
# =========================================================================== #
def score_pair_with_custom_judge(
    gt_text: str,
    pred_text: str,
    judge,
    sample_id: str = "",
    debug_log_path: Path | None = "predictions/judge_logs.json",
) -> "RubricResult":
    from realtext_v2.metrics import RubricResult

    prompt = RUBRIC_PROMPT_TEMPLATE.format(
        gt_text=gt_text or "(empty)",
        pred_text=pred_text or "(empty)",
    )
    raw = judge(prompt)
    parsed = parse_judge_response(raw)

    if debug_log_path is not None:
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sample_id": sample_id,
                "prompt_chars": len(prompt),
                "raw_output": raw,
                "parsed": parsed,
            }, ensure_ascii=False) + "\n")

    return RubricResult(
        factuality=parsed["factuality"],
        reasoning=parsed["reasoning"],
        completeness=parsed["completeness"],
    )

# =========================================================================== #
# Qwen3-VL judge wrapper (defaults to Instruct-32B)
# =========================================================================== #
class QwenJudge:
    """Wraps Qwen3-VL (Instruct or Thinking) as an LLM judge.

    Lazy-loads on first call. Text-only — sees the GT report and the
    candidate report and returns the judge's raw text output.

    Defaults are tuned for the Instruct variant: greedy decoding for
    deterministic structured output, lower max_new_tokens (no chain-of-thought).
    """
    # Defensive: strip any <think>...</think> if the user points the wrapper
    # at a Thinking-family model.
    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        model_id: str,
        model_class: str = "Qwen3VLForConditionalGeneration",
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        attn_impl: str = "sdpa",
        device_map: str = "auto",
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.model_id = model_id
        self.model_class = model_class
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.attn_impl = attn_impl
        self.device_map = device_map
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        import transformers
        from transformers import AutoProcessor

        print(f"[judge] loading {self.model_id} ({self.dtype}) ...")
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        if self._processor.tokenizer.pad_token_id is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token

        ModelCls = getattr(transformers, self.model_class)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        self._model = ModelCls.from_pretrained(
            self.model_id,
            dtype=dtype_map[self.dtype],
            device_map=self.device_map,
            attn_implementation=self.attn_impl,
            trust_remote_code=True,
        ).eval()

    def __call__(self, prompt: str) -> str:
        self._ensure_loaded()
        import torch

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=(self._processor.tokenizer.pad_token_id
                          or self._processor.tokenizer.eos_token_id),
        )
        if self.do_sample:
            gen_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True,
        )[0]
        # Defensive: strip CoT tags if the wrapper is pointed at a Thinking model.
        text = self._THINK_RE.sub("", text).strip()
        return text


# =========================================================================== #
# JSONL helpers (batch mode)
# =========================================================================== #
def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read prediction.jsonl(.gz) into a DataFrame."""
    p = Path(path).expanduser().resolve()
    open_fn = gzip.open if p.suffix == ".gz" else open
    rows = []
    with open_fn(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "image_name" not in df.columns or "report" not in df.columns:
        raise KeyError(
            f"Predictions file must contain 'image_name' and 'report' fields. "
            f"Got: {list(df.columns)}"
        )
    return df


def match_predictions_to_meta(
    preds: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    if "image_file" in meta.columns:
        merged = preds.merge(
            meta, left_on="image_name", right_on="image_file",
            how="inner", suffixes=("", "_meta"),
        )
    else:
        merged = preds.merge(
            meta, left_on="image_name", right_on="sample_id",
            how="inner", suffixes=("", "_meta"),
        )
    if len(merged) == 0:
        raise RuntimeError(
            "No matches between predictions and metadata. "
            f"preds image_name examples: {preds['image_name'].head().tolist()}; "
            f"meta image_file examples: "
            f"{meta.get('image_file', meta['sample_id']).head().tolist()}"
        )
    print(f"[match] preds: {len(preds)}  meta: {len(meta)}  matched: {len(merged)}")
    return merged


def load_predictions_from_dir(preds_dir: str | Path) -> pd.DataFrame:
    """Read per-document ``report.json`` files from a folder of subfolders."""
    root = Path(preds_dir).expanduser().resolve()
    rows = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        json_path = subdir / "report.json"
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append({
            "image_name": data.get("image_name", subdir.name),
            "report": data["report"],
        })
    print(f"[load] {len(rows)} predictions from {len(list(root.iterdir()))} "
          f"subdirs in {root}")
    return pd.DataFrame(rows)


def build_metadata_from_dirs(
    gt_reports_dir: str | Path,
    gt_masks_dir: str | Path,
    images_dir: str | Path,
    image_names: list[str],
) -> pd.DataFrame:
    """Build a metadata table compatible with the batch evaluator.

    Columns: sample_id, image_path, mask_path, report_text, type.
    """
    gt_reports = Path(gt_reports_dir).expanduser().resolve()
    gt_masks = Path(gt_masks_dir).expanduser().resolve()
    images = Path(images_dir).expanduser().resolve()

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    rows = []
    for name in image_names:
        stem = Path(name).stem

        # Image path
        img_path = None
        for ext in _IMG_EXTS:
            candidate = images / f"{stem}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break
        if img_path is None:
            # Try without stripping extension
            for ext in _IMG_EXTS:
                candidate = images / name
                if candidate.exists():
                    img_path = str(candidate)
                    break

        # GT mask
        mask_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = gt_masks / f"{stem}{ext}"
            if candidate.exists():
                mask_path = str(candidate)
                break
        if mask_path is None:
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = gt_masks / f"{stem}_mask{ext}"
                if candidate.exists():
                    mask_path = str(candidate)
                    break

        # GT report
        gt_path = gt_reports / f"{stem}_report.md"
        report_text = ""
        gt_type = "PRISTINE"
        if gt_path.exists():
            report_text = gt_path.read_text(encoding="utf-8")

        # Determine forgery type from report or mask
        from realtext_v2.report import parse_report
        pr = parse_report(report_text) if report_text else None
        if pr and pr.conclusion:
            gt_type = "BLACK" if pr.conclusion.upper() == "FORGED" else "PRISTINE"
        elif mask_path is not None:
            # Fallback: check if mask has any non-zero pixel
            from PIL import Image as PILImage
            try:
                m = PILImage.open(mask_path).convert("L")
                arr = np.array(m, dtype=np.uint8)
                if (arr > 0).any():
                    gt_type = "BLACK"
            except Exception:
                pass

        rows.append({
            "sample_id": stem,
            "image_file": name,
            "image_path": img_path,
            "mask_path": mask_path,
            "report_text": report_text,
            "type": gt_type,
        })

    return pd.DataFrame(rows)


def make_grounding_samples(
    merged: pd.DataFrame,
    *,
    image_h_w: dict[str, tuple[int, int]] | None = None,
    coords_normalized: bool = False,
) -> list:
    from realtext_v2.metrics import GroundingSample
    from realtext_v2.report import parse_report
    from realtext_v2.grounding import boxes_to_mask
    from PIL import Image

    samples = []
    for _, row in merged.iterrows():
        sid = str(row.get("sample_id", row["image_name"]))
        img_path = row.get("image_path")
        gt_mask_path = row.get("mask_path")
        gt_type = str(row.get("type", ""))
        is_forged = gt_type.lower().startswith("black")

        if image_h_w and sid in image_h_w:
            h, w = image_h_w[sid]
        elif img_path is not None and Path(str(img_path)).exists():
            with Image.open(str(img_path)) as im:
                w, h = im.size
        else:
            print(f"  [warn] no image for {sid}; skipping grounding")
            continue

        gt_mask = None
        if gt_mask_path is not None and Path(str(gt_mask_path)).exists():
            m = Image.open(str(gt_mask_path)).convert("L")
            if m.size != (w, h):
                m = m.resize((w, h), Image.NEAREST)
            gt_mask = np.array(m, dtype=np.uint8)
            gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        pred_report_text = str(row["report"])
        pr = parse_report(pred_report_text)
        boxes = []
        for a in pr.anomalies:
            if not a.grounding or len(a.grounding) != 4:
                continue
            x1, y1, x2, y2 = a.grounding
            if coords_normalized:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
        pred_mask = boxes_to_mask(boxes, h, w)

        samples.append(GroundingSample(
            sample_id=sid,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            is_forged=is_forged,
        ))
    return samples


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single-pair mode (judge only, on one GT + one prediction)
    sp = ap.add_argument_group("single-pair judge mode")
    sp.add_argument("--single_gt",
                    help="Path to a single GT report (Markdown text file).")
    sp.add_argument("--single_pred",
                    help="Path to a single candidate report (Markdown text file).")
    sp.add_argument("--single_out", default="single_judge.json",
                    help="Where to write the single-pair judge result.")

    # Batch mode
    bp = ap.add_argument_group("batch mode")
    bp.add_argument("--predictions",
                    help="Path to prediction.jsonl or .jsonl.gz")
    bp.add_argument("--root",
                    help="RealText-V2 dataset root (with metadata.parquet).")
    bp.add_argument("--split_parquet",
                    help="Parquet of the split to evaluate against. "
                         "If omitted, uses ALL of metadata.")
    bp.add_argument("--out", default="eval.json",
                    help="Output JSON file with results (batch mode).")

    # Folder mode (pipeline output with per-document subfolders)
    fp = ap.add_argument_group("folder mode (pipeline output)")
    fp.add_argument("--preds_dir",
                    help="Directory of per-document subfolders, each containing "
                         "report.json.")
    fp.add_argument("--gt_reports_dir",
                    help="Directory of GT Markdown reports ({stem}_report.md).")
    fp.add_argument("--gt_masks_dir",
                    help="Directory of GT binary mask images ({stem}.png etc.).")
    fp.add_argument("--images_dir",
                    help="Directory of original document images (for dimensions).")

    # Component toggles (batch only)
    ap.add_argument("--skip_bertscore", action="store_true")
    ap.add_argument("--skip_rubric", action="store_true")

    # BERTScore options
    ap.add_argument("--bertscore_model",
                    default="xlm-roberta-large")

    # Judge (DEFAULT: Qwen3-VL-32B-Instruct)
    ap.add_argument("--judge_model",
                    default="Qwen/Qwen3-VL-32B-Instruct",
                    help="HF id or local path of the judge model.")
    ap.add_argument("--judge_logs",
                    default="predictions/judge_logs.json")
    ap.add_argument("--judge_class",
                    default="Qwen3VLForConditionalGeneration")
    ap.add_argument("--judge_dtype", default="bfloat16")
    ap.add_argument("--judge_max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_images", type=int, default=100)
    ap.add_argument("--judge_do_sample", action="store_true",
                    help="Enable sampling (for Thinking models). "
                         "Default greedy.")
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_top_p", type=float, default=1.0)
    ap.add_argument("--judge_top_k", type=int, default=0)
    ap.add_argument("--judge_max_samples", type=int, default=0,
                    help="Cap rubric scoring to this many random samples. "
                         "0 = all (slow on 32B).")
    ap.add_argument("--judge_seed", type=int, default=42)

    # Coordinate handling
    ap.add_argument("--coords_normalized", action="store_true",
                    help="Treat predicted [GROUNDING] coords as in [0,1].")

    # Weights for the final score
    ap.add_argument("--weight_det", type=float, default=0.3)
    ap.add_argument("--weight_loc", type=float, default=0.4)
    ap.add_argument("--weight_exp", type=float, default=0.15)
    ap.add_argument("--weight_rep", type=float, default=0.15)

    return ap.parse_args()


def _build_judge(args) -> QwenJudge:
    return QwenJudge(
        args.judge_model,
        model_class=args.judge_class,
        dtype=args.judge_dtype,
        max_new_tokens=args.judge_max_new_tokens,
        do_sample=args.judge_do_sample,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        top_k=args.judge_top_k,
    )


def _read_text(path: str) -> str:
    return Path(path).expanduser().resolve().read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Single-pair mode
# --------------------------------------------------------------------------- #
def run_single_pair(args) -> int:
    gt_text = _read_text(args.single_gt)
    pred_text = _read_text(args.single_pred)
    print(f"[single] GT   : {args.single_gt}  ({len(gt_text)} chars)")
    print(f"[single] PRED : {args.single_pred}  ({len(pred_text)} chars)")

    judge = _build_judge(args)
    t0 = time.time()
    prompt = RUBRIC_PROMPT_TEMPLATE.format(
        gt_text=gt_text or "(empty)",
        pred_text=pred_text or "(empty)",
    )
    raw = judge(prompt)
    parsed = parse_judge_response(raw)
    elapsed = time.time() - t0

    avg = (parsed["factuality"] + parsed["reasoning"]
           + parsed["completeness"]) / 3.0

    print("\n" + "=" * 60)
    print("LLM-JUDGE RUBRIC RESULT")
    print("=" * 60)
    print(f"  factuality   : {parsed['factuality']:6.2f} / 100")
    print(f"  reasoning    : {parsed['reasoning']:6.2f} / 100")
    print(f"  completeness : {parsed['completeness']:6.2f} / 100")
    print(f"  AVERAGE      : {avg:6.2f} / 100  (S_Rep contribution: {avg/100:.4f})")
    print("=" * 60)
    print(f"[done] {elapsed:.1f}s")

    out = {
        "judge_model": args.judge_model,
        "gt_path": str(Path(args.single_gt).resolve()),
        "pred_path": str(Path(args.single_pred).resolve()),
        "factuality": parsed["factuality"],
        "reasoning": parsed["reasoning"],
        "completeness": parsed["completeness"],
        "average": avg,
        "average_norm": avg / 100.0,
        "raw_judge_output": raw,
        "elapsed_sec": elapsed,
    }
    Path(args.single_out).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {args.single_out}")
    return 0


# --------------------------------------------------------------------------- #
# Batch mode
# --------------------------------------------------------------------------- #
def run_batch(args) -> int:
    from realtext_v2 import load_metadata
    from realtext_v2.metadata import resolve_paths
    from realtext_v2.metrics import bertscore_scores as _old_bs
    from bertscore_patch import bertscore_scores_v2 as bertscore_scores
    from realtext_v2.metrics import (
        RubricResult,
        detection_scores,
        explanation_texts_from_report,
        final_score,
        grounding_scores,
    )
    from realtext_v2.report import parse_report

    t0 = time.time()

    if not args.predictions or not args.root:
        raise SystemExit(
            "Batch mode requires --predictions and --root "
            "(or use --single_gt/--single_pred for single-pair mode)."
        )

    # 1. Load predictions
    preds = load_predictions(args.predictions)
    print(f"[load] {len(preds)} predictions from {args.predictions}")

    # 2. Load metadata + resolve paths
    meta = load_metadata(args.root)
    meta = resolve_paths(meta, args.root)
    if args.split_parquet:
        split_meta = pd.read_parquet(args.split_parquet)
        keep_ids = set(split_meta.get(
            "original_sample_id", split_meta.get("sample_id", [])
        ).tolist())
        meta = meta[meta["sample_id"].isin(keep_ids)]
        print(f"[split] restricted to {len(meta)} rows from "
              f"{args.split_parquet}")

    merged = match_predictions_to_meta(preds, meta)

    # 3. Detection (SDet) + parse texts for BERTScore
    print("\n[1/4] Detection ...")
    y_true, y_pred = [], []
    refs_exp, hyps_exp = [], []
    for _, row in merged.iterrows():
        gt = parse_report(str(row.get("report_text", "")))
        pr = parse_report(str(row["report"]))
        y_true.append(
            gt.conclusion if gt.conclusion else
            ("FORGED" if str(row.get("type", "")).lower().startswith("black")
             else "PRISTINE")
        )
        y_pred.append(pr.conclusion or "PRISTINE")
        refs_exp.append(explanation_texts_from_report(gt))
        hyps_exp.append(explanation_texts_from_report(pr))
    det = detection_scores(y_true, y_pred)
    print(f"  precision={det['precision']:.4f}  recall={det['recall']:.4f}  "
          f"f1={det['f1']:.4f}  acc={det['accuracy']:.4f}  n={det['n']}")

    # 4. Grounding (SLoc)
    print("\n[2/4] Grounding ...")
    g_samples = make_grounding_samples(
        merged, coords_normalized=args.coords_normalized,
    )
    loc = grounding_scores(g_samples)
    print(f"  mIoU={loc['mIoU']:.4f}  mF1={loc['mF1']:.4f}  "
          f"mIoU(forged)={loc['mIoU_forged_only']:.4f}  "
          f"mF1(forged)={loc['mF1_forged_only']:.4f}  "
          f"n={loc['n']} ({loc['n_forged']} forged)")

    # 5. Explanation (SExp)
    exp = None
    if not args.skip_bertscore:
        print("\n[3/4] Explanation BERTScore ...")
        try:
            exp = bertscore_scores(
                refs=refs_exp, hyps=hyps_exp,
                model_type=args.bertscore_model,
            )
            print(f"  precision={exp['precision_mean']:.4f}  "
                  f"recall={exp['recall_mean']:.4f}  "
                  f"f1={exp['f1_mean']:.4f}  n={exp['n']}")
        except ImportError:
            print("  [skip] bert-score not installed (pip install bert-score)")
        except Exception as e:  # noqa: BLE001
            print(f"  [error] BERTScore failed: {e}")

    # 6. Rubric (SRep) — custom prompt + Qwen judge
    rep = None
    rubric_results: list[RubricResult] = []
    if not args.skip_rubric:
        print("\n[4/4] LLM-Judge Rubric ...")
        if args.judge_max_samples > 0 and len(merged) > args.judge_max_samples:
            rng = np.random.default_rng(args.judge_seed)
            idx = rng.choice(len(merged), size=args.judge_max_samples,
                             replace=False)
            judged_merged = merged.iloc[idx].reset_index(drop=True)
        else:
            judged_merged = merged

        judge = _build_judge(args)

        for i, (_, row) in enumerate(judged_merged.iterrows(), start=1):
            gt_text = str(row.get("report_text", ""))
            pr_text = str(row["report"])
            try:
                r = score_pair_with_custom_judge(
                    gt_text, pr_text, judge=judge, judge_logs=args.judge_logs
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [judge error] {row.get('sample_id', '?')}: {e}")
                r = RubricResult(0.0, 0.0, 0.0)
            rubric_results.append(r)
            if i % 10 == 0 or i == len(judged_merged):
                avg = np.mean([rr.average for rr in rubric_results])
                print(f"  [{i:4d}/{len(judged_merged)}]  running avg = "
                      f"{avg:.2f}/100")

        if rubric_results:
            avgs = [r.average for r in rubric_results]
            rep = {
                "factuality_mean": float(
                    np.mean([r.factuality for r in rubric_results]) / 100.0
                ),
                "reasoning_mean": float(
                    np.mean([r.reasoning for r in rubric_results]) / 100.0
                ),
                "completeness_mean": float(
                    np.mean([r.completeness for r in rubric_results]) / 100.0
                ),
                "average_norm": float(np.mean(avgs) / 100.0),
                "average":      float(np.mean(avgs)),
                "n":            len(avgs),
            }
            print(f"  Rubric: F={rep['factuality_mean']:.4f}  "
                  f"R={rep['reasoning_mean']:.4f}  "
                  f"C={rep['completeness_mean']:.4f}  "
                  f"avg={rep['average_norm']:.4f}")

    # 7. Final score
    weights = {
        "detection":   args.weight_det,
        "grounding":   args.weight_loc,
        "explanation": args.weight_exp,
        "rubric":      args.weight_rep,
    }
    fs = final_score(det=det, loc=loc, exp=exp, rep=rep, weights=weights)

    out_dict = {
        "predictions_file": str(args.predictions),
        "root":             str(args.root),
        "split_parquet":    str(args.split_parquet) if args.split_parquet else None,
        "judge_model":      args.judge_model,
        "n_predictions":    len(preds),
        "n_matched":        len(merged),
        "weights":          weights,
        "detection":        det,
        "grounding":        loc,
        "explanation":      exp,
        "rubric":           rep,
        "final_score":      float(fs),
        "elapsed_sec":      time.time() - t0,
    }
    Path(args.out).write_text(
        json.dumps(out_dict, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n{'=' * 60}\nFINAL SCORE: {fs:.4f}\n{'=' * 60}")
    print(f"  detection:   {det.get('f1', 0):.4f}")
    print(f"  grounding:   "
          f"{(0.5 * loc.get('mIoU', 0) + 0.5 * loc.get('mF1', 0)):.4f}")
    if exp:
        print(f"  explanation: {exp['f1_mean']:.4f}")
    if rep:
        print(f"  rubric:      {rep['average_norm']:.4f}")
    print(f"\n[done] saved to {args.out}  ({out_dict['elapsed_sec']:.1f}s)")
    return 0


# --------------------------------------------------------------------------- #
# Folder mode (pipeline output with per-document subfolders)
# --------------------------------------------------------------------------- #
def run_folder_mode(args) -> int:
    from realtext_v2.metrics import (
        RubricResult,
        bertscore_scores,
        detection_scores,
        explanation_texts_from_report,
        final_score,
        grounding_scores,
    )
    from realtext_v2.report import parse_report

    t0 = time.time()

    # 1. Load predictions from folder
    preds = load_predictions_from_dir(args.preds_dir)
    if len(preds) == 0:
        print("[error] no report.json files found in subdirs")
        return 1
    
    if args.max_images:
        preds = preds[:args.max_images]

    # 2. Build metadata from GT dirs
    meta = build_metadata_from_dirs(
        args.gt_reports_dir, args.gt_masks_dir, args.images_dir,
        image_names=preds["image_name"].tolist(),
    )
    print(f"[meta] built {len(meta)} metadata rows from GT dirs")

    # 3. Merge
    merged = match_predictions_to_meta(preds, meta)

    # 4. Detection (SDet)
    print("\n[1/4] Detection ...")
    y_true, y_pred = [], []
    refs_exp, hyps_exp = [], []
    for _, row in merged.iterrows():
        gt = parse_report(str(row.get("report_text", "")))
        pr = parse_report(str(row["report"]))
        y_true.append(
            gt.conclusion if gt.conclusion else
            ("FORGED" if str(row.get("type", "")).lower().startswith("black")
             else "PRISTINE")
        )
        y_pred.append(pr.conclusion or "PRISTINE")
        refs_exp.append(explanation_texts_from_report(gt))
        hyps_exp.append(explanation_texts_from_report(pr))
    det = detection_scores(y_true, y_pred)
    print(f"  precision={det['precision']:.4f}  recall={det['recall']:.4f}  "
          f"f1={det['f1']:.4f}  acc={det['accuracy']:.4f}  n={det['n']}")

    # 5. Grounding (SLoc)
    print("\n[2/4] Grounding ...")
    g_samples = make_grounding_samples(
        merged, coords_normalized=args.coords_normalized,
    )
    loc = grounding_scores(g_samples)
    print(f"  mIoU={loc['mIoU']:.4f}  mF1={loc['mF1']:.4f}  "
          f"mIoU(forged)={loc['mIoU_forged_only']:.4f}  "
          f"mF1(forged)={loc['mF1_forged_only']:.4f}  "
          f"n={loc['n']} ({loc['n_forged']} forged)")

    # 6. Explanation (SExp)
    exp = None
    if not args.skip_bertscore:
        print("\n[3/4] Explanation BERTScore ...")
        try:
            exp = bertscore_scores(
                refs=refs_exp, hyps=hyps_exp,
                model_type=args.bertscore_model,
            )
            print(f"  precision={exp['precision_mean']:.4f}  "
                  f"recall={exp['recall_mean']:.4f}  "
                  f"f1={exp['f1_mean']:.4f}  n={exp['n']}")
        except ImportError:
            print("  [skip] bert-score not installed (pip install bert-score)")
        except Exception as e:
            print(f"  [error] BERTScore failed: {e}")

    # 7. Rubric (SRep)
    rep = None
    rubric_results: list[RubricResult] = []
    if not args.skip_rubric:
        print("\n[4/4] LLM-Judge Rubric ...")
        if args.judge_max_samples > 0 and len(merged) > args.judge_max_samples:
            rng = np.random.default_rng(args.judge_seed)
            idx = rng.choice(len(merged), size=args.judge_max_samples,
                             replace=False)
            judged_merged = merged.iloc[idx].reset_index(drop=True)
        else:
            judged_merged = merged

        judge = _build_judge(args)

        for i, (_, row) in enumerate(judged_merged.iterrows(), start=1):
            gt_text = str(row.get("report_text", ""))
            pr_text = str(row["report"])
            try:
                r = score_pair_with_custom_judge(gt_text, pr_text, judge=judge)
            except Exception as e:
                print(f"  [judge error] {row.get('sample_id', '?')}: {e}")
                r = RubricResult(0.0, 0.0, 0.0)
            rubric_results.append(r)
            if i % 10 == 0 or i == len(judged_merged):
                avg = np.mean([rr.average for rr in rubric_results])
                print(f"  [{i:4d}/{len(judged_merged)}]  running avg = "
                      f"{avg:.2f}/100")

        if rubric_results:
            avgs = [r.average for r in rubric_results]
            rep = {
                "factuality_mean": float(
                    np.mean([r.factuality for r in rubric_results]) / 100.0
                ),
                "reasoning_mean": float(
                    np.mean([r.reasoning for r in rubric_results]) / 100.0
                ),
                "completeness_mean": float(
                    np.mean([r.completeness for r in rubric_results]) / 100.0
                ),
                "average_norm": float(np.mean(avgs) / 100.0),
                "average":      float(np.mean(avgs)),
                "n":            len(avgs),
            }
            print(f"  Rubric: F={rep['factuality_mean']:.4f}  "
                  f"R={rep['reasoning_mean']:.4f}  "
                  f"C={rep['completeness_mean']:.4f}  "
                  f"avg={rep['average_norm']:.4f}")

    # 8. Final score
    weights = {
        "detection":   args.weight_det,
        "grounding":   args.weight_loc,
        "explanation": args.weight_exp,
        "rubric":      args.weight_rep,
    }
    fs = final_score(det=det, loc=loc, exp=exp, rep=rep, weights=weights)

    out_dict = {
        "preds_dir":        str(args.preds_dir),
        "gt_reports_dir":   str(args.gt_reports_dir),
        "gt_masks_dir":     str(args.gt_masks_dir),
        "images_dir":       str(args.images_dir),
        "judge_model":      args.judge_model,
        "n_predictions":    len(preds),
        "n_matched":        len(merged),
        "weights":          weights,
        "detection":        det,
        "grounding":        loc,
        "explanation":      exp,
        "rubric":           rep,
        "final_score":      float(fs),
        "elapsed_sec":      time.time() - t0,
    }
    Path(args.out).write_text(
        json.dumps(out_dict, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n{'=' * 60}\nFINAL SCORE: {fs:.4f}\n{'=' * 60}")
    print(f"  detection:   {det.get('f1', 0):.4f}")
    print(f"  grounding:   "
          f"{(0.5 * loc.get('mIoU', 0) + 0.5 * loc.get('mF1', 0)):.4f}")
    if exp:
        print(f"  explanation: {exp['f1_mean']:.4f}")
    if rep:
        print(f"  rubric:      {rep['average_norm']:.4f}")
    print(f"\n[done] saved to {args.out}  ({out_dict['elapsed_sec']:.1f}s)")
    return 0


def main() -> int:
    args = parse_args()
    if args.preds_dir:
        if not (args.gt_reports_dir and args.gt_masks_dir and args.images_dir):
            raise SystemExit(
                "Folder mode requires --preds_dir, --gt_reports_dir, "
                "--gt_masks_dir, and --images_dir."
            )
        return run_folder_mode(args)
    if args.single_gt or args.single_pred:
        if not (args.single_gt and args.single_pred):
            raise SystemExit(
                "Single-pair mode requires BOTH --single_gt and --single_pred."
            )
        return run_single_pair(args)
    return run_batch(args)


if __name__ == "__main__":
    sys.exit(main())