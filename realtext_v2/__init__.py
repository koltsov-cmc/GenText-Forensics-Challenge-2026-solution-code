"""RealText-V2 toolkit: download, inspect, visualize, split, and export
the dataset for VLM supervised fine-tuning.

Dataset: https://huggingface.co/datasets/vankey/RealText-V2
Challenge: https://gentext-forensics-acm-mm-2026.github.io/
"""

from .download import download_dataset, download_metadata_only, DEFAULT_REPO_ID
from .metadata import load_metadata, metadata_stats
from .dataset import RealTextV2Dataset, Sample
from .report import parse_report, serialize_report, ForgeryReport, Anomaly
from .splits import stratified_split
from .viz import plot_sample, plot_grid, save_sample_figure
from .vlm_format import (
    sample_to_chat,
    export_sft_jsonl,
    DEFAULT_USER_PROMPT,
)
from .grounding import (
    boxes_to_mask,
    mask_to_boxes,
    iou as mask_iou,
    pixel_f1 as mask_pixel_f1,
)
from .metrics import (
    detection_scores,
    grounding_scores,
    build_grounding_sample,
    GroundingSample,
    bertscore_scores,
    rubric_score,
    RubricResult,
    evaluate,
    evaluate_predictions_against_meta,
    EvalReport,
    final_score,
    DEFAULT_WEIGHTS,
)

__version__ = "0.2.0"

__all__ = [
    # download / metadata
    "download_dataset",
    "download_metadata_only",
    "DEFAULT_REPO_ID",
    "load_metadata",
    "metadata_stats",
    # dataset
    "RealTextV2Dataset",
    "Sample",
    # report
    "parse_report",
    "serialize_report",
    "ForgeryReport",
    "Anomaly",
    # split + viz
    "stratified_split",
    "plot_sample",
    "plot_grid",
    "save_sample_figure",
    # VLM
    "sample_to_chat",
    "export_sft_jsonl",
    "DEFAULT_USER_PROMPT",
    # grounding
    "boxes_to_mask",
    "mask_to_boxes",
    "mask_iou",
    "mask_pixel_f1",
    # metrics
    "detection_scores",
    "grounding_scores",
    "build_grounding_sample",
    "GroundingSample",
    "bertscore_scores",
    "rubric_score",
    "RubricResult",
    "evaluate",
    "evaluate_predictions_against_meta",
    "EvalReport",
    "final_score",
    "DEFAULT_WEIGHTS",
]
