"""Convert RealText-V2 samples into conversation format for VLM SFT.

The output is a list of JSONL records shaped roughly like the
LLaVA / Qwen-VL / InternVL SFT conventions, but language-framework
agnostic.  Each record looks like::

    {
      "id":       "GenText_Forensic_00000000",
      "image":    "/abs/path/to/image.jpg",
      "mask":     "/abs/path/to/mask.png",
      "language": "en",
      "type":     "black",
      "messages": [
          {"role": "user",      "content": [
              {"type": "image", "image": "<same path>"},
              {"type": "text",  "text":  "<DEFAULT_USER_PROMPT>"}
          ]},
          {"role": "assistant", "content": [
              {"type": "text",  "text":  "<structured markdown report>"}
          ]}
      ]
    }

Adapters for LLaVA-NeXT, Qwen2-VL, InternVL3 etc. can pick up this
format directly or with a tiny reshuffle (e.g. replacing ``<image>``
placeholders for LLaVA).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

import pandas as pd

from .dataset import RealTextV2Dataset, Sample
from .report import ForgeryReport, parse_report, serialize_report


DEFAULT_USER_PROMPT = (
    "You are a multilingual document forensics analyst.  Examine the "
    "attached image and produce a FORGERY ANALYSIS REPORT in Markdown.\n\n"
    "Follow this schema exactly:\n"
    "# FORGERY ANALYSIS REPORT\n"
    "**[Conclusion]:** FORGED | PRISTINE\n"
    "**[RISK_SCORE]:** <0-100>\n\n"
    "For each tampered region, emit:\n"
    "### ANOMALY_<NNN>: <type> (<location>)\n"
    "[GROUNDING]: [x1, y1, x2, y2]\n"
    "[REASON]: <short natural-language justification grounded in visual evidence>\n\n"
    "Conclude with:\n"
    "## SUMMARY\n"
    "<one short paragraph summarising the verdict and the key evidence>\n\n"
    "If the document is authentic, output conclusion PRISTINE, an empty "
    "anomaly list, and a short summary explaining why."
)


def sample_to_chat(
    sample: Sample,
    *,
    user_prompt: str = DEFAULT_USER_PROMPT,
    target_style: Literal["dataset", "submission"] = "dataset",
    image_placeholder: Optional[str] = None,
    include_mask_path: bool = True,
) -> dict:
    """Turn a single ``Sample`` into a chat dict.

    ``image_placeholder`` -- if given (e.g. ``"<image>"``), the user
    text content starts with the placeholder on its own line (LLaVA-
    style).  Otherwise the image is passed as a structured content
    block (Qwen-VL / OpenAI-style).
    """
    report_text: str
    if sample.report_text:
        report_text = sample.report_text
    elif sample.report_path is not None and Path(sample.report_path).exists():
        report_text = Path(sample.report_path).read_text(encoding="utf-8")
    else:
        report_text = ""

    # If a non-default target style is requested, reparse + reserialise.
    if target_style != "dataset" and report_text:
        rep = parse_report(report_text)
        report_text = serialize_report(rep, style=target_style)

    image_path = str(sample.image_path) if sample.image_path else None

    if image_placeholder:
        user_content = f"{image_placeholder}\n{user_prompt}"
    else:
        user_content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_prompt},
        ]

    record: dict = {
        "id": sample.sample_id,
        "image": image_path,
        "language": sample.language_code,
        "type": sample.type,
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": (
                    report_text
                    if image_placeholder
                    else [{"type": "text", "text": report_text}]
                ),
            },
        ],
    }
    if include_mask_path and sample.mask_path is not None:
        record["mask"] = str(sample.mask_path)
    return record


def iter_chat_records(
    ds: RealTextV2Dataset,
    **kw,
) -> Iterator[dict]:
    for s in ds:
        yield sample_to_chat(s, **kw)


def export_sft_jsonl(
    ds: RealTextV2Dataset,
    out_path: str | Path,
    *,
    user_prompt: str = DEFAULT_USER_PROMPT,
    target_style: Literal["dataset", "submission"] = "dataset",
    image_placeholder: Optional[str] = None,
    include_mask_path: bool = True,
    skip_missing_image: bool = True,
) -> int:
    """Dump the dataset to a JSON-Lines file suitable for VLM SFT.

    Returns the number of records written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in ds:
            if skip_missing_image and (s.image_path is None or not Path(s.image_path).exists()):
                continue
            rec = sample_to_chat(
                s,
                user_prompt=user_prompt,
                target_style=target_style,
                image_placeholder=image_placeholder,
                include_mask_path=include_mask_path,
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n
