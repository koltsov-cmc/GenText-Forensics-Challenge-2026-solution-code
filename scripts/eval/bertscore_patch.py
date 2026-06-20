#!/usr/bin/env python
"""Drop-in patch for evaluate_submission.py BERTScore section.

Apply by importing `bertscore_scores_v2` and replacing the call in your
evaluator. Or replace the imported `bertscore_scores` symbol entirely.

The original BERTScore in your code uses `bert-base-multilingual-cased`
which gives very similar embeddings to long forensic reports (precision
plateaus around 0.72-0.76 regardless of content). This patch fixes it:

  1. Default to stronger multilingual model XLM-RoBERTa-large
  2. Chunk long reports before scoring — BERT only sees first 512 tokens
     by default, so long reports give degenerate similarity.
  3. Use rescaled_with_baseline=True so scores spread across [0, 1] range
     instead of clustering near 0.7.

Drop in:
    from bertscore_patch import bertscore_scores_v2
    # Inside evaluator:
    exp = bertscore_scores_v2(refs=..., hyps=..., model_type=args.bertscore_model)
"""
from __future__ import annotations

import re
from typing import Optional


# Default model recommendations (in preference order)
RECOMMENDED_MODELS = {
    # Strong multilingual: covers en, zh, th, ms, id, ar, ru, etc.
    "multilingual_strong": "xlm-roberta-large",
    # Faster multilingual fallback
    "multilingual_fast":   "Alibaba-NLP/gte-multilingual-base",
    # English-only but very discriminative
    "english_strong":      "microsoft/deberta-xlarge-mnli",
    # Old default (do NOT use for long forensic reports)
    "OLD_DEFAULT":         "bert-base-multilingual-cased",
}


def _chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Split long text into chunks of ~max_chars, breaking at sentence
    boundaries when possible. BERTScore's underlying model truncates at
    ~512 tokens (~2000 chars) by default, so chunks of 1000 chars give
    safe coverage of long reports.

    Returns at least one chunk (even if input is empty).
    """
    text = text.strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    # Try to split at section boundaries first (### ANOMALY_, ## SUMMARY, etc.)
    # then sentence boundaries (. ! ?), then just hard split.
    sections = re.split(r"(?=\n#{1,3} )", text)
    sections = [s for s in sections if s.strip()]

    chunks = []
    for sec in sections:
        if len(sec) <= max_chars:
            chunks.append(sec)
        else:
            # Split this section at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", sec)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= max_chars:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                    if len(sent) > max_chars:
                        # Hard split
                        for i in range(0, len(sent), max_chars):
                            chunks.append(sent[i:i+max_chars])
                        current = ""
                    else:
                        current = sent
            if current:
                chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


def bertscore_scores_v2(
    refs: list[str],
    hyps: list[str],
    model_type: str = "xlm-roberta-large",
    rescale_with_baseline: bool = True,
    chunk_long_texts: bool = True,
    chunk_max_chars: int = 1000,
    batch_size: int = 16,
    lang: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """BERTScore with proper handling for long forensic reports.

    Args:
        refs: list of reference (GT) texts
        hyps: list of hypothesis (predicted) texts
        model_type: HF model id. Use xlm-roberta-large for multilingual.
        rescale_with_baseline: spread scores across [0,1] instead of
            clustering near pre-baseline raw cosine values.
        chunk_long_texts: split long texts into chunks and average per-chunk
            BERTScore. Use this for reports > 2000 chars.
        chunk_max_chars: chunk size in characters when chunking enabled.
        batch_size: BERTScore internal batch size for inference.
        lang: language code; only used if rescale_with_baseline=True. For
            multilingual data set to None or "en" (BERTScore baselines are
            per-language but `en` is the most stable for mixed-lang).
        verbose: print progress.

    Returns:
        dict with keys: precision_mean, recall_mean, f1_mean,
        per_sample_precision, per_sample_recall, per_sample_f1, n,
        model_type.
    """
    try:
        from bert_score import BERTScorer
    except ImportError:
        raise ImportError(
            "bert-score not installed. Run: pip install bert-score"
        )

    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) vs hyps ({len(hyps)}) length mismatch")

    if not refs:
        return {
            "precision_mean": 0.0, "recall_mean": 0.0, "f1_mean": 0.0,
            "per_sample_precision": [], "per_sample_recall": [],
            "per_sample_f1": [], "n": 0, "model_type": model_type,
        }

    # Use lang="en" for baseline rescaling because XLM-R's English baseline
    # is the most well-calibrated. For multilingual content it still works
    # better than no rescaling at all.
    effective_lang = lang or "en"

    if verbose:
        print(f"[bertscore] model={model_type}  rescale={rescale_with_baseline}  "
              f"chunk={chunk_long_texts}  lang={effective_lang}")

    scorer = BERTScorer(
        model_type=model_type,
        lang=effective_lang,
        rescale_with_baseline=rescale_with_baseline,
        batch_size=batch_size,
    )

    per_p, per_r, per_f = [], [], []

    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        if chunk_long_texts:
            ref_chunks = _chunk_text(ref, max_chars=chunk_max_chars)
            hyp_chunks = _chunk_text(hyp, max_chars=chunk_max_chars)

            # Align: pad shorter list with empty strings so each chunk is
            # scored against something. This penalises length mismatch
            # appropriately (missing/extra content = empty chunks).
            n_chunks = max(len(ref_chunks), len(hyp_chunks))
            ref_chunks = ref_chunks + [""] * (n_chunks - len(ref_chunks))
            hyp_chunks = hyp_chunks + [""] * (n_chunks - len(hyp_chunks))

            P, R, F = scorer.score(hyp_chunks, ref_chunks)
            # Macro-average over chunks
            per_p.append(float(P.mean()))
            per_r.append(float(R.mean()))
            per_f.append(float(F.mean()))
        else:
            P, R, F = scorer.score([hyp], [ref])
            per_p.append(float(P[0]))
            per_r.append(float(R[0]))
            per_f.append(float(F[0]))

        if verbose and (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(refs)}] running F1: "
                  f"{sum(per_f)/len(per_f):.4f}")

    return {
        "precision_mean": sum(per_p) / len(per_p),
        "recall_mean":    sum(per_r) / len(per_r),
        "f1_mean":        sum(per_f) / len(per_f),
        "per_sample_precision": per_p,
        "per_sample_recall":    per_r,
        "per_sample_f1":        per_f,
        "n":                    len(per_f),
        "model_type":           model_type,
    }


# --------------------------------------------------------------------------- #
# How to patch evaluate_submission.py
# --------------------------------------------------------------------------- #
PATCH_INSTRUCTIONS = """
To apply this fix to your existing evaluate_submission.py:

1. Save bertscore_patch.py next to evaluate_submission.py.

2. In evaluate_submission.py, replace the bertscore_scores import:
       OLD:
           from realtext_v2.metrics import bertscore_scores
       NEW:
           from realtext_v2.metrics import bertscore_scores as _old_bs
           from bertscore_patch import bertscore_scores_v2 as bertscore_scores

3. Change the default --bertscore_model:
       OLD:
           default="bert-base-multilingual-cased"
       NEW:
           default="xlm-roberta-large"

4. (Optional) Add CLI flags for the new behaviour:
       ap.add_argument("--bertscore_no_chunk", action="store_true")
       ap.add_argument("--bertscore_no_rescale", action="store_true")

   Then pass:
       exp = bertscore_scores(
           refs=refs_exp, hyps=hyps_exp,
           model_type=args.bertscore_model,
           chunk_long_texts=not args.bertscore_no_chunk,
           rescale_with_baseline=not args.bertscore_no_rescale,
       )
"""

if __name__ == "__main__":
    print(PATCH_INSTRUCTIONS)