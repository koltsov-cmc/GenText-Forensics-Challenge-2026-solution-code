"""Parse and serialise the structured forgery-analysis markdown reports.

The dataset uses this schema (see RealText-V2 README)::

    # FORGERY ANALYSIS REPORT

    **[Conclusion]:** FORGED | PRISTINE
    **[RISK_SCORE]:** 0-100

    ### ANOMALY_001: <type> (<location>)
    [GROUNDING]: [x1, y1, x2, y2]
    [REASON]: <free text>

    ### ANOMALY_002: ...

    ## SUMMARY
    <free text>

The evaluation server (per challenge.html) expects the final submission
in a slightly different layout -- see ``serialize_report(..., style="submission")``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional


CONCLUSION_FORGED = "FORGED"
CONCLUSION_PRISTINE = "PRISTINE"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Anomaly:
    index: int
    type: str = ""
    location: str = ""
    grounding: Optional[List[int]] = None    # [x1, y1, x2, y2] (pixel coords)
    reason: str = ""
    forged_content: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForgeryReport:
    conclusion: str = CONCLUSION_PRISTINE                # FORGED / PRISTINE
    risk_score: Optional[int] = None                     # 0-100
    anomalies: List[Anomaly] = field(default_factory=list)
    summary: str = ""
    raw: str = ""

    @property
    def is_forged(self) -> bool:
        return self.conclusion.upper().startswith("FORG")

    def to_dict(self) -> dict:
        return {
            "conclusion": self.conclusion,
            "risk_score": self.risk_score,
            "summary": self.summary,
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
_RE_CONCLUSION = re.compile(
    r"\[Conclusion\]\**\s*:?\**\s*([A-Za-z_\-/ ]+)", re.IGNORECASE
)
_RE_RISK = re.compile(
    r"\[RISK[_ ]SCORE\]\**\s*:?\**\s*(\d+)", re.IGNORECASE
)
_RE_ANOMALY_HDR = re.compile(
    r"^#{2,4}\s*ANOMALY_?(\d+)\s*:?\s*(.*?)\s*$", re.MULTILINE
)
_RE_GROUNDING = re.compile(
    r"\[GROUNDING\]\s*:?\s*\[?\s*([0-9.,\s\-]+?)\s*\]?\s*(?:$|\n)",
    re.IGNORECASE,
)
_RE_REASON = re.compile(
    r"\[REASON\]\s*:?\s*(.+?)(?=\n\s*\[[A-Z_]+\]|\n\s*#{2,}|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_RE_CONTENT = re.compile(
    r"\[(?:FORGED_CONTENT|CONTENT|TEXT)\]\s*:?\s*(.+?)(?=\n\s*\[[A-Z_]+\]|\n\s*#{2,}|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_RE_SUMMARY = re.compile(
    r"^#{1,3}\s*SUMMARY\s*\n(.*?)(?=\n#{1,3}\s|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_RE_HDR_LOCATION = re.compile(r"\((.*?)\)\s*$")


def parse_report(text: str) -> ForgeryReport:
    """Parse a RealText-V2 report string into a ``ForgeryReport``.

    The parser is permissive: missing fields yield empty defaults, and
    malformed bounding boxes are silently dropped rather than raising.
    """
    rep = ForgeryReport(raw=text)

    # Conclusion
    m = _RE_CONCLUSION.search(text)
    if m:
        rep.conclusion = m.group(1).strip().upper().replace(" ", "_")
        if "FORG" in rep.conclusion:
            rep.conclusion = CONCLUSION_FORGED
        elif "PRIST" in rep.conclusion or "AUTH" in rep.conclusion or "REAL" in rep.conclusion:
            rep.conclusion = CONCLUSION_PRISTINE

    # Risk score
    m = _RE_RISK.search(text)
    if m:
        try:
            rep.risk_score = int(m.group(1))
        except ValueError:
            rep.risk_score = None

    # Anomaly blocks. Split on the anomaly headers so each block can be
    # parsed independently.
    header_matches = list(_RE_ANOMALY_HDR.finditer(text))
    for i, hm in enumerate(header_matches):
        start = hm.end()
        end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)
        # Don't swallow the SUMMARY block into the last anomaly.
        summary_m = _RE_SUMMARY.search(text, hm.end())
        if summary_m and summary_m.start() < end:
            end = summary_m.start()
        block = text[start:end]

        anomaly = Anomaly(index=int(hm.group(1)))

        hdr_tail = hm.group(2).strip()
        loc_m = _RE_HDR_LOCATION.search(hdr_tail)
        if loc_m:
            anomaly.location = loc_m.group(1).strip()
            anomaly.type = hdr_tail[: loc_m.start()].strip(" :\t")
        else:
            anomaly.type = hdr_tail

        gm = _RE_GROUNDING.search(block)
        if gm:
            coords = [
                c.strip() for c in gm.group(1).replace(";", ",").split(",")
                if c.strip()
            ]
            try:
                anomaly.grounding = [int(round(float(c))) for c in coords[:4]]
                if len(anomaly.grounding) != 4:
                    anomaly.grounding = None
            except ValueError:
                anomaly.grounding = None

        rm = _RE_REASON.search(block)
        if rm:
            anomaly.reason = rm.group(1).strip()

        cm = _RE_CONTENT.search(block)
        if cm:
            anomaly.forged_content = cm.group(1).strip()

        rep.anomalies.append(anomaly)

    # Summary
    sm = _RE_SUMMARY.search(text)
    if sm:
        rep.summary = sm.group(1).strip()

    return rep


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def serialize_report(
    report: ForgeryReport,
    *,
    style: Literal["dataset", "submission"] = "dataset",
) -> str:
    """Serialise a ``ForgeryReport`` back to markdown.

    ``style="dataset"`` reproduces the training-data layout.
    ``style="submission"`` follows the challenge.html submission schema
    (Region N: [x1, y1, x2, y2], content).
    """
    if style == "dataset":
        return _serialize_dataset(report)
    if style == "submission":
        return _serialize_submission(report)
    raise ValueError(f"Unknown style: {style}")


def _serialize_dataset(r: ForgeryReport) -> str:
    lines = ["# FORGERY ANALYSIS REPORT", ""]
    lines.append(f"**[Conclusion]:** {r.conclusion}")
    score = 0 if r.risk_score is None else r.risk_score
    lines.append(f"**[RISK_SCORE]:** {score}")
    lines.append("")
    for a in r.anomalies:
        loc = f" ({a.location})" if a.location else ""
        typ = a.type or "unknown"
        lines.append(f"### ANOMALY_{a.index:03d}: {typ}{loc}")
        if a.grounding:
            lines.append(f"[GROUNDING]: [{', '.join(str(x) for x in a.grounding)}]")
        if a.forged_content:
            lines.append(f"[FORGED_CONTENT]: {a.forged_content}")
        if a.reason:
            lines.append(f"[REASON]: {a.reason}")
        lines.append("")
    lines.append("## SUMMARY")
    lines.append(r.summary or "")
    return "\n".join(lines).rstrip() + "\n"


def _serialize_submission(r: ForgeryReport) -> str:
    verdict = "Forged" if r.is_forged else "Authentic"
    lines = [
        f"Conclusion: {verdict}",
        f"Score: {0 if r.risk_score is None else r.risk_score}",
        "",
        "Tamper Regions:",
    ]
    if not r.anomalies:
        lines.append("(none)")
    for i, a in enumerate(r.anomalies, start=1):
        bbox = a.grounding if a.grounding else [0, 0, 0, 0]
        content = a.forged_content or a.reason or ""
        content = content.replace("\n", " ")
        lines.append(f"Region {i}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}], {content}")
    if r.summary:
        lines.append("")
        lines.append("Summary:")
        lines.append(r.summary)
    return "\n".join(lines) + "\n"
