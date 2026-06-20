"""Grounding utilities: conversions between bounding boxes and binary masks.

The challenge's grounding metric operates on pixel-level masks (TruFor
protocol).  The model, however, outputs bounding boxes inside the
markdown report.  These helpers close that gap.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def boxes_to_mask(
    boxes: Iterable[Sequence[int]],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterise a list of ``[x1, y1, x2, y2]`` boxes into a uint8 mask.

    Coordinates are clipped to the image bounds; degenerate / empty
    boxes are ignored.  Output has values in {0, 255} to match the
    native RealText-V2 mask format.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in boxes or []:
        if b is None or len(b) < 4:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in b[:4])
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def mask_to_boxes(mask: np.ndarray, min_area: int = 16) -> list[list[int]]:
    """Extract axis-aligned bounding boxes around connected components
    of a binary mask.  Uses a simple flood-fill; no OpenCV dependency.
    """
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return []

    h, w = bin_mask.shape
    seen = np.zeros_like(bin_mask, dtype=bool)
    boxes: list[list[int]] = []

    for start_y in range(h):
        for start_x in range(w):
            if not bin_mask[start_y, start_x] or seen[start_y, start_x]:
                continue
            # Iterative DFS to find the bounding box of this component.
            stack = [(start_y, start_x)]
            y_min, y_max = start_y, start_y
            x_min, x_max = start_x, start_x
            area = 0
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= h or x < 0 or x >= w:
                    continue
                if seen[y, x] or not bin_mask[y, x]:
                    continue
                seen[y, x] = True
                area += 1
                if y < y_min: y_min = y
                if y > y_max: y_max = y
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                stack.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
            if area >= min_area:
                boxes.append([x_min, y_min, x_max + 1, y_max + 1])
    return boxes


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Binary mask IoU. Both inputs must be the same shape."""
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        # Both empty -> perfect match (by convention).
        return 1.0
    return float(inter) / float(union)


def pixel_f1(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Pixel-level F1 between two binary masks."""
    a = mask_a > 0
    b = mask_b > 0
    tp = np.logical_and(a, b).sum()
    fp = np.logical_and(~a, b).sum()
    fn = np.logical_and(a, ~b).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp) / float(denom)


def boxes_from_report_grounding(
    anomalies,
    height: int,
    width: int,
) -> list[list[int]]:
    """Convenience: pull bbox list out of a parsed report's anomalies."""
    boxes: list[list[int]] = []
    for a in anomalies or []:
        g = getattr(a, "grounding", None)
        if g and len(g) == 4:
            boxes.append([int(v) for v in g])
    return boxes
