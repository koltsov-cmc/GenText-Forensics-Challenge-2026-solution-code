#!/usr/bin/env python
"""Pre-render student prompts to disk from CoT JSON files.

Reads cot.json files and substitutes {{OCR_JSON}}, {{DTD_HINTS}}, {{FILTERED_DTD}}
into prompt templates, producing one .prompt.txt file per sample for each
template (stage1, stage2, direct, etc.).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Anchors / patterns
# --------------------------------------------------------------------------- #
_STAGE1_ANCHOR_RE = re.compile(
    r"---\s*STAGE\s*1\s*[:.\-]\s*Knowledge\s*Preparation\s*---", re.IGNORECASE)
_STAGE2_ANCHOR_RE = re.compile(r"---\s*STAGE\s*2\s*[:.\-].*?---", re.IGNORECASE)
_STAGE3_ANCHOR_RE = re.compile(r"---\s*STAGE\s*3\s*[:.\-].*?---", re.IGNORECASE)

# A REGION block: starts with REGION_NNN, runs across newlines until the
# "→ KEEP/DROP ..." verdict line ends with a period.
_REGION_BLOCK_RE = re.compile(
    r"(REGION_\d+\b.*?\u2192\s*(?:KEEP|DROP)\b[^\n.]*\.)",
    re.DOTALL | re.IGNORECASE,
)
_KEEP_VERDICT_RE = re.compile(
    r"\u2192\s*KEEP\s+as\s+(?:Semantic\s+Subtle|Visual\s+Clumsy|Logical\s+Fraud)\s*\.\s*$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# OCR helpers
# --------------------------------------------------------------------------- #
def _parse_ocr_input(ocr_input_str: str) -> dict:
    """Parse `ocr_input` JSON string into a dict. Returns {} on failure."""
    if not isinstance(ocr_input_str, str) or not ocr_input_str.strip():
        return {}
    try:
        return json.loads(ocr_input_str)
    except json.JSONDecodeError:
        return {}


def _format_ocr_compact(ocr_input_str: str) -> str:
    """Convert verbose ocr_input JSON into a compact triplet string PLUS a
    reading-order text block at the end.

    Returned format:
        Detected words as (text, [bbox], conf) triplets:
        ("a", [...], 0.99), ("b", [...], 0.97), ...

        Reading-order text:
        <full reading_order_text>
    """
    blob = _parse_ocr_input(ocr_input_str)
    if not blob:
        return "No OCR text detected."

    items = blob.get("ocr_items", []) or []
    reading_order = (blob.get("reading_order_text") or "").strip()

    if not items and not reading_order:
        return "No OCR text detected."

    parts = []
    if items:
        triplets = []
        for it in items:
            text = it.get("text", "")
            bbox = it.get("bbox", [])
            conf = it.get("confidence", 0.0)
            triplets.append(f'("{text}", {bbox}, {conf})')
        parts.append(
            "Detected words as (text, [bbox], confidence) triplets:\n\n"
            + ", ".join(triplets)
        )
    if reading_order:
        parts.append("Reading-order text:\n\n" + reading_order)

    return "\n\n".join(parts)


def _bbox_iou_or_overlap_frac(box_a, box_b) -> float:
    """Return the fraction of box_b's area covered by box_a (overlap fraction
    relative to box_b). Used to decide if an OCR item "is inside" a DTD region.
    Returns 0.0 if no overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    # Order
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


# --------------------------------------------------------------------------- #
# DTD hints formatting
# --------------------------------------------------------------------------- #
def _format_dtd_hints_with_overlap(
    dtd_regions: list,
    ocr_items: list,
    min_overlap_frac: float = 0.4,
    max_overlapping_items: int = 6,
) -> str:
    """Format dtd_regions as numbered region lines with OCR-overlap info.

    Each line shows the region's bbox and the OCR words whose own bbox is
    at least `min_overlap_frac` covered by the region. This gives the
    student a verbal anchor for what's inside each region instead of pure
    pixel arithmetic.
    """
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
            frac = _bbox_iou_or_overlap_frac((x1, y1, x2, y2), ocr_bbox)
            if frac >= min_overlap_frac:
                overlaps.append((frac, it))

        # Sort by overlap fraction desc, take top N
        overlaps.sort(key=lambda kv: -kv[0])
        overlaps = overlaps[:max_overlapping_items]

        if overlaps:
            overlap_strs = []
            for frac, it in overlaps:
                tid = it.get("id", "?")
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


# --------------------------------------------------------------------------- #
# FILTERED_DTD construction (stage 1 text + KEEP regions)
# --------------------------------------------------------------------------- #
def _trim_tail(body: str) -> str:
    """Strip trailing junk markers (</think>, </tool_call>, ``` etc.)."""
    tail_patterns = (
        re.compile(r"</\s*think\s*>\s*$",     re.IGNORECASE),
        re.compile(r"</\s*tool_call\s*>\s*$", re.IGNORECASE),
        re.compile(r"<\s*think\s*>\s*$",      re.IGNORECASE),
        re.compile(r"<\s*tool_call\s*>\s*$",  re.IGNORECASE),
        re.compile(r"```(?:\w+)?\s*$"),
        re.compile(r"\*\*\s*END\s+OF\s+REPORT\s*\*\*\s*$", re.IGNORECASE),
    )
    body = body.rstrip()
    for _ in range(20):
        stripped = False
        for pat in tail_patterns:
            m_tail = pat.search(body)
            if m_tail is not None:
                body = body[: m_tail.start()].rstrip()
                stripped = True
                break
        if not stripped:
            break
    return body


def _extract_stage1_text(raw_output: str) -> Optional[str]:
    """Extract STAGE 1 text from STAGE 1 anchor up to (but excluding) STAGE 2
    anchor. If STAGE 2 is missing, go to STAGE 3 or end-of-text."""
    m1 = _STAGE1_ANCHOR_RE.search(raw_output)
    if not m1:
        return None
    start = m1.start()
    m2 = _STAGE2_ANCHOR_RE.search(raw_output)
    if m2 and m2.start() > start:
        end = m2.start()
    else:
        m3 = _STAGE3_ANCHOR_RE.search(raw_output)
        end = m3.start() if m3 else len(raw_output)
    body = _trim_tail(raw_output[start:end])
    return body if body else None


def _extract_keep_only_blocks(raw_output: str) -> list[str]:
    """Find all REGION_NNN blocks in STAGE 2 and return only those ending
    with 'KEEP as <Type>.'. Drops every DROP block."""
    m2 = _STAGE2_ANCHOR_RE.search(raw_output)
    if not m2:
        return []
    m3 = _STAGE3_ANCHOR_RE.search(raw_output)
    end = m3.start() if m3 else len(raw_output)
    stage2_text = raw_output[m2.start():end]

    kept = []
    for m in _REGION_BLOCK_RE.finditer(stage2_text):
        block = m.group(1).strip()
        if _KEEP_VERDICT_RE.search(block):
            kept.append(block)
    return kept


def _build_filtered_dtd(raw_output: str) -> str:
    """Build {{FILTERED_DTD}} = STAGE 1 text + KEEP region blocks.

    Returns a placeholder string if the teacher output is missing pieces,
    so downstream pipeline always has SOMETHING to substitute.
    """
    stage1 = _extract_stage1_text(raw_output)
    kept_blocks = _extract_keep_only_blocks(raw_output)

    pieces: list[str] = []
    if stage1:
        pieces.append(stage1)

    pieces.append("--- STAGE 2 KEEP regions (filtered) ---")
    if kept_blocks:
        pieces.append("\n\n".join(kept_blocks))
    else:
        pieces.append("No confirmed tampering regions.")

    if not stage1 and not kept_blocks:
        # If we have nothing at all, prefix with explicit note
        pieces.insert(0, "(STAGE 1 unavailable)")

    return "\n\n".join(pieces)


# --------------------------------------------------------------------------- #
# Per-template substitution
# --------------------------------------------------------------------------- #
def _substitute(template_str: str, template_stem: str, rec: dict) -> str:
    """Apply the right substitutions for the given template."""
    raw_output = rec.get("raw_teacher_output", "") or ""
    ocr_input_str = rec.get("ocr_input", "") or ""
    dtd_regions = rec.get("dtd_regions") or []

    blob = _parse_ocr_input(ocr_input_str)
    ocr_items = blob.get("ocr_items", []) or []

    if template_stem == "student_prompt_stage1":
        # Only DTD_HINTS; no OCR
        out = template_str.replace(
            "{{DTD_HINTS}}",
            _format_dtd_hints_with_overlap(dtd_regions, ocr_items),
        )
        # Defensive: scrub OCR_JSON slot if accidentally present
        out = out.replace("{{OCR_JSON}}", "")
        return out

    if template_stem == "student_prompt_stage2":
        # OCR + FILTERED_DTD; no DTD_HINTS, no heatmap
        out = (
            template_str
            .replace("{{OCR_JSON}}",  _format_ocr_compact(ocr_input_str))
            .replace("{{FILTERED_DTD}}", _build_filtered_dtd(raw_output))
        )
        out = out.replace("{{DTD_HINTS}}", "")
        return out

    # Default: prompt_direct_v4 / prompt_strong_direct / student_prompt_v4
    return (
        template_str
        .replace("{{OCR_JSON}}",  _format_ocr_compact(ocr_input_str))
        .replace("{{DTD_HINTS}}",
                 _format_dtd_hints_with_overlap(dtd_regions, ocr_items))
    )


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def _gather_cot_files(cot_root: Path) -> list[Path]:
    out: list[Path] = []
    for part_dir in sorted(cot_root.iterdir()):
        if part_dir.is_dir() and part_dir.name.lower().startswith("part"):
            out.extend(sorted(part_dir.glob("*.cot.json")))
    return out


def _load_template(prompts_dir: Path, name: str) -> tuple[str, str]:
    """Return (stem, template_text). 'name' may be a bare filename or a path."""
    p = Path(name)
    if not p.is_absolute() and not p.exists():
        p = prompts_dir / name
    if not p.exists():
        raise SystemExit(f"prompt not found: {p}")
    return p.stem, p.read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cot_root",    required=True)
    ap.add_argument("--prompts_dir", required=True,
                    help="Directory containing the prompt .txt files.")
    ap.add_argument("--out_root",    required=True,
                    help="Outputs go to {out_root}/{prompt_stem}/{partXXX}/{stem}.prompt.txt")
    ap.add_argument(
        "--prompts", nargs="*",
        default=[
            "prompt_direct_v4.txt",
            "prompt_strong_direct.txt",
            "student_prompt_v4.txt",
            "student_prompt_stage1.txt",
            "student_prompt_stage2.txt",
        ],
        help="Prompt filenames (relative to --prompts_dir) to render.",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cot_root = Path(args.cot_root).expanduser().resolve()
    prompts_dir = Path(args.prompts_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not cot_root.is_dir():
        raise SystemExit(f"cot_root not a directory: {cot_root}")
    if not prompts_dir.is_dir():
        raise SystemExit(f"prompts_dir not a directory: {prompts_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    templates = []
    for name in args.prompts:
        stem, text = _load_template(prompts_dir, name)
        templates.append((stem, text))
        print(f"[prompt] {stem}  ({len(text)} chars)", flush=True)

    cot_files = _gather_cot_files(cot_root)
    print(f"[scan] {len(cot_files)} cot.json files under {cot_root}", flush=True)
    if args.limit > 0:
        cot_files = cot_files[: args.limit]
        print(f"[limit] processing first {len(cot_files)}", flush=True)

    counters = {f"ok_{s}": 0 for s, _ in templates}
    counters["skip_exists"] = 0
    counters["decode_err"] = 0
    counters["exception"]  = 0

    for i, cot_path in enumerate(cot_files, start=1):
        part = cot_path.parent.name
        stem = cot_path.name[: -len(".cot.json")] \
               if cot_path.name.endswith(".cot.json") else cot_path.stem
        try:
            rec = json.loads(cot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counters["decode_err"] += 1
            continue

        for tpl_stem, tpl_text in templates:
            out_path = out_root / tpl_stem / part / f"{stem}.prompt.txt"
            if out_path.exists() and not args.overwrite:
                counters["skip_exists"] += 1
                continue
            try:
                rendered = _substitute(tpl_text, tpl_stem, rec)
            except Exception as exc:
                counters["exception"] += 1
                print(f"  [error] {tpl_stem}/{part}/{stem}: {exc!r}",
                      flush=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered, encoding="utf-8")
            counters[f"ok_{tpl_stem}"] += 1

        if i % 200 == 0:
            print(f"  [{i}/{len(cot_files)}] {counters}", flush=True)

    print(f"\n[done] {counters}")
    print(f"[done] outputs in {out_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())