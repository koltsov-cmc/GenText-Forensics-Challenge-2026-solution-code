# RealText-V2 toolkit

Tooling for the [ACM MM 2026 GenText-Forensics challenge](https://gentext-forensics-acm-mm-2026.github.io/)
([Codabench](https://www.codabench.org/competitions/15805/)).

Implements the **third-place** method: a two-stage CoT pipeline with DTD visual
forgery detection and two LoRA-adapted Qwen3-VL-32B models for forensic reasoning
and report generation.

## Method overview

1. **DTD** — statistical visual forgery detector, produces tampering probability maps
2. **CoT distillation** — Qwen3-VL-235B teacher generates chain-of-thought from GT data
3. **SFT training** — two LoRA-adapted Qwen3-VL-32B models trained on distilled CoT
4. **Two-stage inference** — Stage 1 filters DTD regions, Stage 2 generates the
   forensic report with OCR grounding

## Install

```bash
pip install -e .
```

## Package layout

```
realtext_v2_toolkit/
├── realtext_v2/              Python package (dataset, reports, splits, viz)
├── prompts/for_all/          Prompt templates for teacher/student
├── DTD_TRAIN.yaml            DTD training configuration
├── pyproject.toml
├── README.md
└── scripts/
    ├── dtd/                  DTD detection and OCR extraction
    │   ├── extract_dtd_ocr.py
    │   ├── extract_ocr.py
    │   └── extract_regions.py
    ├── cot/                  Chain-of-thought distillation and prep
    │   ├── distill_cot.py
    │   ├── prerender_prompts.py
    │   └── augment_reports.py
    ├── training/
    │   └── train_student.py  LoRA SFT training
    ├── inference/            Pipeline inference
    │   ├── infer_student.py
    │   ├── infer_student_sharded.py
    │   ├── run_pipeline.py
    │   ├── run_combined.py
    │   ├── qwen3vl.py
    │   ├── paddle_ocr.py
    │   └── paddle_ocr_subprocess.py
    ├── eval/                 Evaluation metrics
    │   ├── evaluate_submission.py
    │   ├── eval_stage1.py
    │   ├── eval_dtd_thresholds.py
    │   ├── eval_dtd_on_val.py
    │   ├── eval_stage1_on_val.py
    │   └── bertscore_patch.py
    └── utils/                Utilities
        ├── annotate_heatmaps.py
        ├── vis_report.py
        ├── viz_dtd_boxes.py
        ├── prompt_stats.py
        └── download_hf.py
```

## Quick inference (single image)

```bash
python scripts/inference/qwen3vl.py --image /path/to/doc.jpg --out_dir predictions/
```

## Licence

Code: MIT. Dataset: CC-BY-NC-4.0 (research only), per the RealText-V2 card.
