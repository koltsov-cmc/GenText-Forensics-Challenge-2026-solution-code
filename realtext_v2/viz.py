"""Visualisation helpers for RealText-V2 samples.

All functions work headlessly (you can ``save=True``) and in notebooks.
They depend only on matplotlib + PIL + numpy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .dataset import Sample


MASK_ALPHA = 0.45
BBOX_COLOR_FORGED = "#ff2e2e"
BBOX_COLOR_OTHER = "#ffb800"


# --------------------------------------------------------------------------- #
def _overlay_mask(ax, image: Image.Image, mask: np.ndarray, alpha: float = MASK_ALPHA):
    ax.imshow(image)
    if mask is None:
        return
    # Build an RGBA overlay: red where mask > 0.
    h, w = mask.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    m = mask > 0
    overlay[m, 0] = 1.0    # R
    overlay[m, 3] = alpha  # A
    ax.imshow(overlay)


def _draw_bboxes(ax, anomalies, image_size):
    w, h = image_size
    for a in anomalies:
        if not a.grounding or len(a.grounding) != 4:
            continue
        x1, y1, x2, y2 = a.grounding
        rect = mpatches.Rectangle(
            (x1, y1),
            max(1, x2 - x1),
            max(1, y2 - y1),
            linewidth=2,
            edgecolor=BBOX_COLOR_FORGED,
            facecolor="none",
        )
        ax.add_patch(rect)
        label = f"A{a.index}"
        if a.type:
            label += f" · {a.type[:20]}"
        ax.text(
            x1,
            max(0, y1 - 4),
            label,
            color="white",
            fontsize=8,
            bbox=dict(
                facecolor=BBOX_COLOR_FORGED,
                edgecolor="none",
                alpha=0.9,
                pad=1.5,
            ),
        )


def _truncate(txt: str, n: int = 600) -> str:
    txt = txt.strip()
    return txt if len(txt) <= n else txt[: n - 1] + "…"


# --------------------------------------------------------------------------- #
def plot_sample(
    sample: Sample,
    *,
    show_report: bool = True,
    figsize: tuple[float, float] = (14, 7),
    max_report_chars: int = 900,
) -> plt.Figure:
    """Render one sample: image + mask-overlay + parsed report side-by-side."""
    img = sample.image()
    mask = sample.mask()
    report = sample.report() if (sample.report_text or sample.report_path) else None

    n_cols = 3 if show_report else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    ax_img, ax_mask = axes[0], axes[1]
    ax_img.imshow(img)
    ax_img.set_title(f"{sample.sample_id} · {sample.language_code} · {sample.type}")
    ax_img.axis("off")
    if report is not None:
        _draw_bboxes(ax_img, report.anomalies, img.size)

    _overlay_mask(ax_mask, img, mask)
    ax_mask.set_title("Mask overlay" if mask is not None else "No mask (pristine)")
    ax_mask.axis("off")

    if show_report:
        ax_txt = axes[2]
        ax_txt.axis("off")
        if report is None:
            ax_txt.text(0, 0.5, "(no report)", fontsize=10)
        else:
            lines = [
                f"Conclusion: {report.conclusion}",
                f"Risk score: {report.risk_score}",
                f"Anomalies:  {len(report.anomalies)}",
                "",
            ]
            for a in report.anomalies[:6]:
                bbox = a.grounding if a.grounding else "-"
                lines.append(f"[{a.index}] {a.type or '?'}  {bbox}")
                if a.reason:
                    lines.append(f"    {_truncate(a.reason, 180)}")
            if len(report.anomalies) > 6:
                lines.append(f"...and {len(report.anomalies) - 6} more")
            if report.summary:
                lines.append("")
                lines.append("SUMMARY:")
                lines.append(_truncate(report.summary, max_report_chars))
            ax_txt.text(
                0.0,
                1.0,
                "\n".join(lines),
                fontsize=9,
                family="monospace",
                verticalalignment="top",
                wrap=True,
            )
        ax_txt.set_title("Report")

    fig.tight_layout()
    return fig


def plot_grid(
    samples: Sequence[Sample],
    *,
    ncols: int = 4,
    figsize_per_cell: tuple[float, float] = (3.2, 3.2),
    show_mask: bool = True,
) -> plt.Figure:
    """Render many samples as a compact grid for exploration."""
    n = len(samples)
    if n == 0:
        raise ValueError("empty sample list")
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )
    for i, s in enumerate(samples):
        ax = axes[i // ncols][i % ncols]
        try:
            img = s.image()
            if show_mask:
                _overlay_mask(ax, img, s.mask())
            else:
                ax.imshow(img)
            if s.report_text or s.report_path:
                try:
                    report = s.report()
                    _draw_bboxes(ax, report.anomalies, img.size)
                except Exception:
                    pass
        except Exception as e:
            ax.text(0.5, 0.5, f"error: {e}", ha="center", va="center")
        ax.set_title(
            f"{s.sample_id}\n{s.language_code} · {s.type}", fontsize=8
        )
        ax.axis("off")
    # Hide unused cells.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    return fig


def save_sample_figure(sample: Sample, out_path: str | Path, **kw) -> Path:
    """Render ``plot_sample`` and save to disk (PNG)."""
    fig = plot_sample(sample, **kw)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
