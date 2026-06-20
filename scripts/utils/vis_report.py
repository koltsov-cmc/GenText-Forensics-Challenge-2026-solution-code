"""Visualize a forensic report next to its image.

For each anomaly in the report:
  * Draw a red bbox on the image with a short label (#NNN  Type).
  * Place the [REASON] paragraph in the LEFT or RIGHT side margin (whichever
    side is opposite the bbox's horizontal half — so text never sits on top
    of its own region of interest).
  * Connect the bbox to its text block with a thin leader line.

Vertical position of each text block is the centre of the corresponding bbox,
with a simple top-down packing pass to avoid overlapping blocks on the same
side.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.parse_math"] = False 
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # needed for np.ndarray in mask handling
from PIL import Image  # noqa: E402


# --------------------------------------------------------------------------- #
# Tiny inline parser (avoids depending on the toolkit's parse_report)
# --------------------------------------------------------------------------- #
_ANOMALY_HEADER_RE = re.compile(
    r"###\s*ANOMALY_?(\d+)\s*:?\s*([^\n]*)",
    re.IGNORECASE,
)
_GROUNDING_RE = re.compile(
    r"\[GROUNDING\]\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)
_REASON_RE = re.compile(
    r"\[REASON\]\s*:\s*(.+?)"
    r"(?=\n\s*###\s*ANOMALY"
    r"|\n\s*---"
    r"|\n\s*##\s"
    r"|\n\s*\*\*END\s+OF\s+REPORT\*\*"
    r"|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_anomalies(report_text: str) -> list[dict]:
    """Extract anomalies as [{index, type, bbox, reason}, ...]."""
    parts = list(_ANOMALY_HEADER_RE.finditer(report_text))
    anomalies: list[dict] = []
    for i, m in enumerate(parts):
        start = m.start()
        end = parts[i + 1].start() if i + 1 < len(parts) else len(report_text)
        chunk = report_text[start:end]

        idx_str = m.group(1)
        type_label = m.group(2).strip().rstrip(":")

        gm = _GROUNDING_RE.search(chunk)
        if not gm:
            continue
        bbox = [int(gm.group(j)) for j in (1, 2, 3, 4)]

        rm = _REASON_RE.search(chunk)
        reason = rm.group(1).strip() if rm else ""
        reason = " ".join(reason.split())  # collapse internal whitespace

        anomalies.append({
            "index": idx_str,
            "type": type_label,
            "bbox": bbox,
            "reason": reason,
        })
    return anomalies


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _bbox_iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def visualize_report_with_mask(
    image_path: str | Path,
    report_path: str | Path,
    out_path: str | Path,
    *,
    gt_mask_path: str | Path,
    text_column_width_ratio: float = 0.40,
    text_max_chars_per_line: int = 55,
    text_fontsize: int = 8,
    title: Optional[str] = None,
    bbox_color: str = "#ff2e2e",
    target_dpi: int = 150,
) -> Path:
    """Like :func:`visualize_report`, but bboxes come from the GT mask.

    Each connected component in the mask is matched (by IoU) to the
    closest report anomaly.  The mask-derived bbox is drawn on the
    image while the report's type / reason text stays in the side
    panels.  The mapping is 1:1 — every mask component pairs with
    exactly one anomaly.
    """
    image_path = Path(image_path)
    report_path = Path(report_path)
    out_path = Path(out_path)
    gt_mask_path = Path(gt_mask_path)

    # Parse report
    report_text = report_path.read_text(encoding="utf-8")
    anomalies = _parse_anomalies(report_text)

    # Load GT mask, extract connected-component bboxes
    mask_pil = Image.open(str(gt_mask_path)).convert("L")
    mask_arr = np.array(mask_pil, dtype=np.uint8)
    mask_arr = (mask_arr > 0).astype(np.uint8) * 255

    try:
        from realtext_v2.grounding import mask_to_boxes
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from realtext_v2.grounding import mask_to_boxes
    mask_bboxes = mask_to_boxes(mask_arr, min_area=10)
    if not mask_bboxes:
        # No mask regions — fall back to regular viz
        return visualize_report(
            image_path, report_path, out_path,
            text_column_width_ratio=text_column_width_ratio,
            text_max_chars_per_line=text_max_chars_per_line,
            text_fontsize=text_fontsize, title=title,
            bbox_color=bbox_color, target_dpi=target_dpi,
        )

    # Match each anomaly to the best-overlapping mask bbox (max IoU)
    for a in anomalies:
        best_iou = -1.0
        best_box = None
        for mb in mask_bboxes:
            iou = _bbox_iou(a["bbox"], list(mb))
            if iou > best_iou:
                best_iou = iou
                best_box = list(mb)
        if best_box is not None and best_iou > 0.0:
            a["bbox"] = best_box   # replace with mask-derived bbox

    # ---- Rendering (same as visualize_report) ----
    image = Image.open(str(image_path)).convert("RGB")
    W_img, H_img = image.size

    if not anomalies:
        # No anomalies in report — still render the image
        _render_simple(image, out_path, title or "No anomalies", target_dpi)
        return out_path

    text_col_w = text_column_width_ratio * W_img
    total_w_data = W_img + 2 * text_col_w

    fig_h_in = max(7.0, min(14.0, H_img / target_dpi))
    fig_w_in = fig_h_in * total_w_data / H_img
    fig_w_in = max(14.0, min(32.0, fig_w_in))
    fig_h_in = fig_w_in * H_img / total_w_data

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=target_dpi)
    ax.set_xlim(-text_col_w, W_img + text_col_w)
    ax.set_ylim(H_img, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    ax.imshow(image, extent=[0, W_img, H_img, 0])

    def _measure_line_height_data() -> float:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        n_test = 10
        test_str = "\n".join(["Sample"] * n_test)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        sample = ax.text(
            x0 - 10 * (x1 - x0),
            y0 + 10 * (y0 - y1),
            test_str, fontsize=text_fontsize, va="top",
        )
        fig.canvas.draw()
        ext = sample.get_window_extent(renderer)
        inv = ax.transData.inverted()
        y_top_data = inv.transform((0, ext.y1))[1]
        y_bot_data = inv.transform((0, ext.y0))[1]
        sample.remove()
        return abs(y_top_data - y_bot_data) / n_test

    line_height_data = _measure_line_height_data() * 1.05
    block_padding_data = line_height_data * 0.7

    left_blocks, right_blocks = [], []
    for a in anomalies:
        x1, y1, x2, y2 = a["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        header_str = f"#{a['index']}  {a['type']}".strip()
        wrapped = textwrap.wrap(a["reason"], width=text_max_chars_per_line) or [""]
        n_lines = 1 + len(wrapped)
        block_h = n_lines * line_height_data

        block = {
            "anomaly": a,
            "header": header_str,
            "wrapped_lines": wrapped,
            "block_h": block_h,
            "target_y": cy,
            "bbox_cx": cx,
            "bbox_cy": cy,
        }
        if cx >= W_img / 2:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    def _layout(blocks: list[dict]) -> float:
        if not blocks:
            return 0.0
        blocks.sort(key=lambda b: b["target_y"])
        y_cursor = 0.0
        for b in blocks:
            ideal = b["target_y"] - b["block_h"] / 2
            y_top = max(ideal, y_cursor)
            y_top = max(0.0, y_top)
            b["y_top"] = y_top
            b["y_bot"] = y_top + b["block_h"]
            y_cursor = b["y_bot"] + block_padding_data
        return max(b["y_bot"] for b in blocks)

    max_y_left = _layout(left_blocks)
    max_y_right = _layout(right_blocks)
    effective_h = max(float(H_img), max_y_left, max_y_right) + block_padding_data

    if effective_h > H_img:
        ax.set_ylim(effective_h, 0)
        new_fig_h = fig_w_in * effective_h / total_w_data
        fig.set_size_inches(fig_w_in, new_fig_h)
        fig.canvas.draw()

    # Draw bboxes
    for a in anomalies:
        x1, y1, x2, y2 = a["bbox"]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        rect = mpatches.Rectangle(
            (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
            linewidth=2, edgecolor=bbox_color, facecolor="none",
        )
        ax.add_patch(rect)
        short_type = re.split(r"\s*\(", a["type"], maxsplit=1)[0].strip()
        label = f"#{a['index']}"
        if short_type:
            label += f"  {short_type[:32]}"
        ax.text(
            x1, max(0, y1 - 6), label,
            color="white", fontsize=text_fontsize + 1, fontweight="bold",
            bbox=dict(facecolor=bbox_color, edgecolor="none", alpha=0.9, pad=2),
            zorder=5,
        )

    def _draw_block(b: dict, side: str) -> None:
        if side == "left":
            x_text = -text_col_w + 0.03 * text_col_w
            line_anchor_x = 0
            line_anchor_text_x = -0.01 * text_col_w
        else:
            x_text = W_img + 0.03 * text_col_w
            line_anchor_x = W_img
            line_anchor_text_x = W_img + 0.01 * text_col_w

        panel = mpatches.FancyBboxPatch(
            (x_text - 0.015 * text_col_w, b["y_top"] - line_height_data * 0.25),
            text_col_w * 0.92,
            b["block_h"] + line_height_data * 0.4,
            boxstyle="round,pad=0.2",
            linewidth=0.6, edgecolor="#cccccc", facecolor="#fafafa",
            alpha=0.9, zorder=2,
        )
        ax.add_patch(panel)
        ax.text(
            x_text, b["y_top"], b["header"],
            ha="left", va="top",
            fontsize=text_fontsize + 1, fontweight="bold",
            color=bbox_color, zorder=3,
        )
        body = "\n".join(b["wrapped_lines"])
        ax.text(
            x_text, b["y_top"] + line_height_data * 1.05, body,
            ha="left", va="top",
            fontsize=text_fontsize, color="#222222",
            family="sans-serif", zorder=3,
        )
        x1, y1, x2, y2 = b["anomaly"]["bbox"]
        bbox_y_mid = (y1 + y2) / 2
        if side == "left":
            line_start = (x1, bbox_y_mid)
        else:
            line_start = (x2, bbox_y_mid)
        line_end = (line_anchor_text_x, b["y_top"] + b["block_h"] / 2)
        ax.annotate(
            "", xy=line_end, xytext=line_start,
            arrowprops=dict(
                arrowstyle="-", color=bbox_color, alpha=0.55, linewidth=0.9,
            ),
            zorder=4,
        )

    for b in left_blocks:
        _draw_block(b, "left")
    for b in right_blocks:
        _draw_block(b, "right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=target_dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return out_path


def _render_simple(
    image: Image.Image,
    out_path: Path,
    title: str,
    target_dpi: int,
) -> None:
    """Render just the image with a title when there are no anomalies."""
    W_img, H_img = image.size
    fig_h_in = max(7.0, min(14.0, H_img / target_dpi))
    fig_w_in = fig_h_in * W_img / H_img
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=target_dpi)
    ax.imshow(image)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=target_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
def visualize_report(
    image_path: str | Path,
    report_path: str | Path,
    out_path: str | Path,
    *,
    text_column_width_ratio: float = 0.40,
    text_max_chars_per_line: int = 55,
    text_fontsize: int = 8,
    title: Optional[str] = None,
    bbox_color: str = "#ff2e2e",
    target_dpi: int = 150,
) -> Path:
    """Render image + bboxes + per-anomaly REASON paragraphs in side margins.

    Parameters
    ----------
    image_path
        Path to the source image.
    report_path
        Path to the forensic report (Markdown text file). Anomalies are
        extracted via the standard ``### ANOMALY_NNN`` / ``[GROUNDING]:`` /
        ``[REASON]:`` schema.
    out_path
        Where to save the resulting PNG.
    text_column_width_ratio
        Width of each side column as a fraction of the image's pixel width.
    text_max_chars_per_line
        Word-wrap column for the REASON text.
    text_fontsize
        Base font size for REASON text. Headers render one point larger.
    title
        Optional title above the figure.
    bbox_color
        Colour of bboxes, header banners, and leader lines.
    target_dpi
        DPI used both for figure sizing and for `savefig`.
    """
    image_path = Path(image_path)
    report_path = Path(report_path)
    out_path = Path(out_path)

    image = Image.open(str(image_path)).convert("RGB")
    W_img, H_img = image.size

    report_text = report_path.read_text(encoding="utf-8")
    anomalies = _parse_anomalies(report_text)

    text_col_w = text_column_width_ratio * W_img
    total_w_data = W_img + 2 * text_col_w

    # Initial figure size: matched to image aspect, with side margins included.
    fig_h_in = max(7.0, min(14.0, H_img / target_dpi))
    fig_w_in = fig_h_in * total_w_data / H_img
    fig_w_in = max(14.0, min(32.0, fig_w_in))
    fig_h_in = fig_w_in * H_img / total_w_data

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=target_dpi)
    ax.set_xlim(-text_col_w, W_img + text_col_w)
    ax.set_ylim(H_img, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    ax.imshow(image, extent=[0, W_img, H_img, 0])

    # --- Measure ACTUAL per-line height in data units via renderer -----------
    # Approximate calculations from fig_h_in * dpi don't account for title bars,
    # axes padding, or aspect=equal letterboxing, so we render a real sample
    # text and read its extent back through transData.inverted().
    def _measure_line_height_data() -> float:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        n_test = 10
        test_str = "\n".join(["Sample"] * n_test)
        # Place WAY outside the visible viewport so it never appears in output.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()  # y0 > y1 because inverted
        sample = ax.text(
            x0 - 10 * (x1 - x0),
            y0 + 10 * (y0 - y1),
            test_str, fontsize=text_fontsize, va="top",
        )
        fig.canvas.draw()
        ext = sample.get_window_extent(renderer)
        inv = ax.transData.inverted()
        y_top_data = inv.transform((0, ext.y1))[1]
        y_bot_data = inv.transform((0, ext.y0))[1]
        sample.remove()
        return abs(y_top_data - y_bot_data) / n_test

    line_height_data = _measure_line_height_data() * 1.05  # 5% safety margin
    block_padding_data = line_height_data * 0.7

    # --- Wrap text, decide a column for each anomaly, compute block heights --
    left_blocks, right_blocks = [], []
    for a in anomalies:
        x1, y1, x2, y2 = a["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        header_str = f"#{a['index']}  {a['type']}".strip()
        wrapped = textwrap.wrap(a["reason"], width=text_max_chars_per_line) or [""]
        n_lines = 1 + len(wrapped)  # header + reason
        block_h = n_lines * line_height_data

        block = {
            "anomaly": a,
            "header": header_str,
            "wrapped_lines": wrapped,
            "block_h": block_h,
            "target_y": cy,
            "bbox_cx": cx,
            "bbox_cy": cy,
        }
        # bbox on RIGHT half of image -> annotate on LEFT side; vice versa.
        if cx >= W_img / 2:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    # --- If the side text stack is taller than the image, GROW the figure ----
    # We add empty canvas below the image so panels can extend past H_img
    # without overlapping each other.
    #
    # The layout pass intentionally has NO bottom cap — collision avoidance
    # via y_cursor guarantees blocks never overlap, and we then expand the
    # figure to accommodate whatever stack height resulted.
    def _layout(blocks: list[dict]) -> float:
        if not blocks:
            return 0.0
        blocks.sort(key=lambda b: b["target_y"])
        y_cursor = 0.0
        for b in blocks:
            ideal = b["target_y"] - b["block_h"] / 2
            y_top = max(ideal, y_cursor)
            y_top = max(0.0, y_top)
            b["y_top"] = y_top
            b["y_bot"] = y_top + b["block_h"]
            y_cursor = b["y_bot"] + block_padding_data
        return max(b["y_bot"] for b in blocks)

    max_y_left = _layout(left_blocks)
    max_y_right = _layout(right_blocks)
    effective_h = max(float(H_img), max_y_left, max_y_right) + block_padding_data

    if effective_h > H_img:
        ax.set_ylim(effective_h, 0)
        new_fig_h = fig_w_in * effective_h / total_w_data
        fig.set_size_inches(fig_w_in, new_fig_h)
        # transData has changed; line_height_data stays valid (it's in data
        # units and aspect=equal preserves 1 data unit = 1 image pixel).
        fig.canvas.draw()

    # --- Draw bboxes on the image with their type label ---
    for a in anomalies:
        x1, y1, x2, y2 = a["bbox"]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        rect = mpatches.Rectangle(
            (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
            linewidth=2, edgecolor=bbox_color, facecolor="none",
        )
        ax.add_patch(rect)

        # Short type for the bbox label (drop parenthetical specifics).
        short_type = re.split(r"\s*\(", a["type"], maxsplit=1)[0].strip()
        label = f"#{a['index']}"
        if short_type:
            label += f"  {short_type[:32]}"
        ax.text(
            x1, max(0, y1 - 6), label,
            color="white", fontsize=text_fontsize + 1,
            fontweight="bold",
            bbox=dict(facecolor=bbox_color, edgecolor="none", alpha=0.9, pad=2),
            zorder=5,
        )

    # --- Draw side text blocks + leader lines ---
    def _draw_block(b: dict, side: str) -> None:
        if side == "left":
            x_text = -text_col_w + 0.03 * text_col_w
            line_anchor_x = 0  # leader line ends at the image's left edge
            line_anchor_text_x = -0.01 * text_col_w
        else:
            x_text = W_img + 0.03 * text_col_w
            line_anchor_x = W_img
            line_anchor_text_x = W_img + 0.01 * text_col_w

        # Background panel behind the text block (helps readability).
        panel = mpatches.FancyBboxPatch(
            (x_text - 0.015 * text_col_w, b["y_top"] - line_height_data * 0.25),
            text_col_w * 0.92,
            b["block_h"] + line_height_data * 0.4,
            boxstyle="round,pad=0.2",
            linewidth=0.6,
            edgecolor="#cccccc",
            facecolor="#fafafa",
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(panel)

        # Header.
        ax.text(
            x_text, b["y_top"], b["header"],
            ha="left", va="top",
            fontsize=text_fontsize + 1, fontweight="bold",
            color=bbox_color,
            zorder=3,
        )

        # Body (reason).
        body = "\n".join(b["wrapped_lines"])
        ax.text(
            x_text, b["y_top"] + line_height_data * 1.05, body,
            ha="left", va="top",
            fontsize=text_fontsize, color="#222222",
            family="sans-serif",
            zorder=3,
        )

        # Leader line: from the bbox edge nearest this side to the panel edge,
        # at the bbox's vertical centre on one end and the block's centre on
        # the other.
        x1, y1, x2, y2 = b["anomaly"]["bbox"]
        bbox_y_mid = (y1 + y2) / 2
        if side == "left":
            line_start = (x1, bbox_y_mid)
        else:
            line_start = (x2, bbox_y_mid)
        line_end = (line_anchor_text_x, b["y_top"] + b["block_h"] / 2)
        ax.annotate(
            "", xy=line_end, xytext=line_start,
            arrowprops=dict(
                arrowstyle="-",
                color=bbox_color,
                alpha=0.55,
                linewidth=0.9,
            ),
            zorder=4,
        )

    for b in left_blocks:
        _draw_block(b, "left")
    for b in right_blocks:
        _draw_block(b, "right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=target_dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--col_ratio", type=float, default=0.40)
    ap.add_argument("--max_chars", type=int, default=55)
    ap.add_argument("--fontsize", type=int, default=8)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    out = visualize_report(
        args.image, args.report, args.out,
        text_column_width_ratio=args.col_ratio,
        text_max_chars_per_line=args.max_chars,
        text_fontsize=args.fontsize,
        title=args.title,
    )
    print(f"saved -> {out}")