"""Evaluation metrics for the GenText-Forensics / RealText-V2 Track 1.

Four components (from challenge.html):

    * Detection (SDet)    -- Binary F1 on Forged/Pristine classification.
    * Grounding (SLoc)    -- Pixel-level mean F1 + mean IoU between
                             predicted and ground-truth tamper masks
                             (TruFor protocol).
    * Explanation (SExp)  -- BERTScore F1 between predicted and
                             ground-truth `[REASON]` / SUMMARY text.
    * Report Rubric (SRep)-- Rubric-based LLM-Judge score (Factuality,
                             Reasoning, Completeness), normalised 0-1.

The final score is a weighted sum; the exact official weights (S_fin.png
on the challenge site) are left configurable.  Defaults are 0.25 each.

Everything is dependency-light: numpy + sklearn only for the core; BERT-
score and the LLM judge are lazy-imported so the rest of the module can
be used without those dependencies installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

from .grounding import boxes_to_mask, iou, pixel_f1
from .report import CONCLUSION_FORGED, ForgeryReport, parse_report


# --------------------------------------------------------------------------- #
# Detection (SDet)
# --------------------------------------------------------------------------- #
def detection_scores(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    positive_label: str = CONCLUSION_FORGED,
) -> dict:
    """Binary detection metrics.

    ``y_true`` / ``y_pred`` are sequences of conclusion strings
    (e.g. ``"FORGED"``, ``"PRISTINE"``).
    """
    yt = np.array([1 if str(v).upper().startswith("FORG") else 0 for v in y_true])
    yp = np.array([1 if str(v).upper().startswith("FORG") else 0 for v in y_pred])
    return {
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "accuracy": float((yt == yp).mean()),
        "n": int(len(yt)),
    }


# --------------------------------------------------------------------------- #
# Grounding (SLoc)
# --------------------------------------------------------------------------- #
@dataclass
class GroundingSample:
    sample_id: str
    gt_mask: Optional[np.ndarray]        # uint8, {0,255}; None for pristine
    pred_mask: np.ndarray                 # uint8, {0,255}
    is_forged: bool                       # per GT

    @property
    def iou(self) -> float:
        gt = self.gt_mask if self.gt_mask is not None else np.zeros_like(self.pred_mask)
        return iou(gt, self.pred_mask)

    @property
    def pixel_f1(self) -> float:
        gt = self.gt_mask if self.gt_mask is not None else np.zeros_like(self.pred_mask)
        return pixel_f1(gt, self.pred_mask)


def grounding_scores(samples: Iterable[GroundingSample]) -> dict:
    """Aggregate per-sample grounding metrics.

    Returns mIoU and mF1 over all samples (pristine samples contribute
    perfect scores iff the prediction is empty -- aligning with TruFor).
    We also split scores over forged-only for a more informative view.
    """
    all_iou, all_f1, forg_iou, forg_f1 = [], [], [], []
    for s in samples:
        all_iou.append(s.iou)
        all_f1.append(s.pixel_f1)
        if s.is_forged:
            forg_iou.append(s.iou)
            forg_f1.append(s.pixel_f1)
    out = {
        "mIoU": float(np.mean(all_iou)) if all_iou else 0.0,
        "mF1": float(np.mean(all_f1)) if all_f1 else 0.0,
        "mIoU_forged_only": float(np.mean(forg_iou)) if forg_iou else 0.0,
        "mF1_forged_only": float(np.mean(forg_f1)) if forg_f1 else 0.0,
        "n": len(all_iou),
        "n_forged": len(forg_iou),
    }
    return out


def build_grounding_sample(
    *,
    sample_id: str,
    gt_mask_path: Optional[str | Path],
    image_size_hw: tuple[int, int],
    pred_report: ForgeryReport,
    is_forged: bool,
) -> GroundingSample:
    """Rasterise the predicted bounding boxes into a mask at the image
    resolution, then pair with the ground-truth mask for evaluation."""
    h, w = image_size_hw
    pred_boxes = [a.grounding for a in pred_report.anomalies if a.grounding]
    pred_mask = boxes_to_mask(pred_boxes, h, w)

    gt_mask: Optional[np.ndarray] = None
    if gt_mask_path is not None and Path(gt_mask_path).exists():
        m = Image.open(gt_mask_path).convert("L")
        gt_mask = np.array(m, dtype=np.uint8)
        # Resize to the image resolution if mismatched.
        if gt_mask.shape != (h, w):
            m = m.resize((w, h), resample=Image.NEAREST)
            gt_mask = np.array(m, dtype=np.uint8)
    return GroundingSample(
        sample_id=sample_id,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        is_forged=is_forged,
    )


# --------------------------------------------------------------------------- #
# Explanation (SExp)
# --------------------------------------------------------------------------- #
def explanation_texts_from_report(report: ForgeryReport) -> str:
    """Concatenate REASON + SUMMARY fields into a single string for
    semantic similarity scoring."""
    parts: list[str] = []
    for a in report.anomalies:
        if a.reason:
            parts.append(a.reason.strip())
    if report.summary:
        parts.append(report.summary.strip())
    return "\n".join(parts).strip()


def bertscore_scores(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    model_type: str = "bert-base-multilingual-cased",
    batch_size: int = 16,
    device: Optional[str] = None,
    lang: Optional[str] = None,
) -> dict:
    """Compute BERTScore precision / recall / F1 between parallel lists.

    Lazy-imports ``bert_score`` -- install via ``pip install bert-score``.
    """
    from bert_score import score as _bert_score  # lazy

    assert len(refs) == len(hyps), "refs and hyps length mismatch"
    # Replace empty strings with a single space to avoid library error.
    refs_safe = [r if r.strip() else " " for r in refs]
    hyps_safe = [h if h.strip() else " " for h in hyps]

    P, R, F = _bert_score(
        cands=hyps_safe,
        refs=refs_safe,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        lang=lang,
        verbose=False,
    )
    return {
        "precision_mean": float(P.mean()),
        "recall_mean": float(R.mean()),
        "f1_mean": float(F.mean()),
        "per_sample_f1": [float(x) for x in F.tolist()],
        "n": len(refs),
    }


# --------------------------------------------------------------------------- #
# Report Quality Rubrics (SRep)   -- LLM Judge hook
# --------------------------------------------------------------------------- #
RUBRIC_PROMPT_TEMPLATE = """\
You are an expert document-forgery forensic judge. Score the CANDIDATE
report against the GROUND-TRUTH report on three 0-100 axes:

1. Factuality   -- accuracy of the verdict and cited evidence.
2. Reasoning    -- logical deduction from visual clues.
3. Completeness -- coverage of manipulation regions and schema compliance.

Return ONLY a compact JSON object of the form:
{{"factuality": <int>, "reasoning": <int>, "completeness": <int>}}

GROUND-TRUTH:
---
{ground_truth}
---

CANDIDATE:
---
{candidate}
---
"""


@dataclass
class RubricResult:
    factuality: float
    reasoning: float
    completeness: float

    @property
    def average(self) -> float:
        return (self.factuality + self.reasoning + self.completeness) / 3.0


def rubric_score(
    gt_report: str,
    pred_report: str,
    *,
    llm_judge: Callable[[str], str],
    prompt_template: str = RUBRIC_PROMPT_TEMPLATE,
) -> RubricResult:
    """Score a single (gt, pred) pair via an LLM judge.

    ``llm_judge`` is a pluggable callable that takes a prompt and
    returns a JSON-ish string.  Example wrappers below.
    """
    import json
    import re
    prompt = prompt_template.format(ground_truth=gt_report, candidate=pred_report)
    raw = llm_judge(prompt)
    # Extract the first {...} block robustly.
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return RubricResult(0.0, 0.0, 0.0)
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return RubricResult(0.0, 0.0, 0.0)
    return RubricResult(
        factuality=float(d.get("factuality", 0)),
        reasoning=float(d.get("reasoning", 0)),
        completeness=float(d.get("completeness", 0)),
    )


def mock_llm_judge(prompt: str) -> str:  # noqa: ARG001
    """A zero-dependency judge used for smoke testing only."""
    return '{"factuality": 75, "reasoning": 70, "completeness": 65}'


# --------------------------------------------------------------------------- #
# Final aggregated score
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {
    "detection": 0.25,
    "grounding": 0.25,
    "explanation": 0.25,
    "rubric": 0.25,
}


@dataclass
class EvalReport:
    detection: dict = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)
    explanation: Optional[dict] = None
    rubric: Optional[dict] = None
    final: float = 0.0
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> dict:
        return {
            "detection": self.detection,
            "grounding": self.grounding,
            "explanation": self.explanation,
            "rubric": self.rubric,
            "weights": self.weights,
            "final": self.final,
        }


def final_score(
    *,
    det: dict,
    loc: dict,
    exp: Optional[dict] = None,
    rep: Optional[dict] = None,
    weights: Optional[dict] = None,
) -> float:
    """Combine the four components into a single scalar in [0, 1].

    Component values are taken from:
      * det["f1"]
      * loc["mIoU"]       (could be replaced by 0.5*mIoU + 0.5*mF1)
      * exp["f1_mean"]
      * rep["average_norm"]   (0-1 normalised)

    Missing components have their weight redistributed to the rest.
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    comps: dict[str, float] = {"detection": det.get("f1", 0.0)}
    if loc:
        comps["grounding"] = 0.5 * loc.get("mIoU", 0.0) + 0.5 * loc.get("mF1", 0.0)
    if exp is not None:
        comps["explanation"] = exp.get("f1_mean", 0.0)
    if rep is not None:
        comps["rubric"] = rep.get("average_norm", rep.get("average", 0.0))

    # Redistribute missing weights to keep the formula in [0, 1].
    total_w = sum(w[k] for k in comps)
    if total_w <= 0:
        return 0.0
    return float(sum(w[k] * comps[k] for k in comps) / total_w)


def evaluate(
    *,
    y_true_conclusion: Sequence[str],
    y_pred_conclusion: Sequence[str],
    grounding_samples: Optional[Iterable[GroundingSample]] = None,
    refs_exp: Optional[Sequence[str]] = None,
    hyps_exp: Optional[Sequence[str]] = None,
    rubric_results: Optional[Sequence[RubricResult]] = None,
    weights: Optional[dict] = None,
    bertscore_kwargs: Optional[dict] = None,
) -> EvalReport:
    """High-level one-shot evaluation.  Skip any component by passing None."""
    out = EvalReport(weights=dict(weights or DEFAULT_WEIGHTS))
    out.detection = detection_scores(y_true_conclusion, y_pred_conclusion)

    if grounding_samples is not None:
        out.grounding = grounding_scores(grounding_samples)

    if refs_exp is not None and hyps_exp is not None:
        out.explanation = bertscore_scores(refs_exp, hyps_exp, **(bertscore_kwargs or {}))

    if rubric_results is not None:
        avgs = [r.average for r in rubric_results]
        out.rubric = {
            "factuality_mean": float(np.mean([r.factuality for r in rubric_results]) / 100.0),
            "reasoning_mean": float(np.mean([r.reasoning for r in rubric_results]) / 100.0),
            "completeness_mean": float(np.mean([r.completeness for r in rubric_results]) / 100.0),
            "average_norm": float(np.mean(avgs) / 100.0),
            "average": float(np.mean(avgs)),
            "n": len(avgs),
        }

    out.final = final_score(
        det=out.detection,
        loc=out.grounding,
        exp=out.explanation,
        rep=out.rubric,
        weights=out.weights,
    )
    return out


# --------------------------------------------------------------------------- #
# High-level helpers for predictions saved as JSONL
# --------------------------------------------------------------------------- #
def evaluate_predictions_against_meta(
    *,
    preds: pd.DataFrame,                  # cols: sample_id, pred_report (str)
    meta: pd.DataFrame,                   # resolved metadata with image_path, mask_path, report_text
    judge: Optional[Callable[[str], str]] = None,
    weights: Optional[dict] = None,
    bertscore_kwargs: Optional[dict] = None,
    skip_bertscore: bool = False,
) -> EvalReport:
    """Evaluate a predictions DataFrame against an (already-resolved)
    metadata DataFrame.  Both must contain ``sample_id``.
    """
    merged = preds.merge(meta, on="sample_id", how="inner", suffixes=("_pred", ""))
    if "report_text" not in merged.columns:
        raise KeyError("metadata must include report_text (ground truth)")

    y_true, y_pred = [], []
    refs_exp, hyps_exp = [], []
    grounding_samples: list[GroundingSample] = []
    rubric_results: list[RubricResult] = []

    for _, row in merged.iterrows():
        gt = parse_report(row["report_text"])
        pr = parse_report(row["pred_report"])
        y_true.append(gt.conclusion)
        y_pred.append(pr.conclusion)
        refs_exp.append(explanation_texts_from_report(gt))
        hyps_exp.append(explanation_texts_from_report(pr))

        img_path = row.get("image_path")
        if img_path is not None and Path(str(img_path)).exists():
            with Image.open(str(img_path)) as im:
                w, h = im.size
            grounding_samples.append(
                build_grounding_sample(
                    sample_id=str(row["sample_id"]),
                    gt_mask_path=row.get("mask_path"),
                    image_size_hw=(h, w),
                    pred_report=pr,
                    is_forged=gt.is_forged,
                )
            )
        if judge is not None:
            rubric_results.append(rubric_score(row["report_text"], row["pred_report"], llm_judge=judge))

    return evaluate(
        y_true_conclusion=y_true,
        y_pred_conclusion=y_pred,
        grounding_samples=grounding_samples or None,
        refs_exp=None if skip_bertscore else refs_exp,
        hyps_exp=None if skip_bertscore else hyps_exp,
        rubric_results=rubric_results or None,
        weights=weights,
        bertscore_kwargs=bertscore_kwargs,
    )
