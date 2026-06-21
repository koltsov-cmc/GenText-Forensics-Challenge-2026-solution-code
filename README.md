# RealText-V2 toolkit

Tooling for the [ACM MM 2026 GenText-Forensics challenge](https://gentext-forensics-acm-mm-2026.github.io/)
([Codabench](https://www.codabench.org/competitions/15805/)).

Implements the **third-place** method: a two-stage CoT pipeline with DTD visual
forgery detection and two LoRA-adapted Qwen3-VL-32B models for forensic reasoning
and report generation.

<img width="3632" height="1318" alt="dtd_example" src="https://github.com/user-attachments/assets/357b6417-586a-4856-9f4c-df86594e5b63" />


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
    │   ├── run_two_stage_hf.py  Self-contained end-to-end runner (HF assets)
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

## Run the full pipeline from scratch (single image)

`scripts/inference/run_two_stage_hf.py` is a self-contained, end-to-end runner.
It downloads every required asset from the public Hugging Face dataset repo
[`cmcshnik/GenText-Forensics_third_place_additional_materials`](https://huggingface.co/datasets/cmcshnik/GenText-Forensics_third_place_additional_materials)
— the DTD weights + model code (`dtd.pth`, `dtd_backbones/`), and the two LoRA
adapters (`qwen_filterer/`, `qwen_semantic_detective/`) — fetches the
`Qwen/Qwen3-VL-32B-Instruct` base model, and runs DTD → OCR → Filterer →
Semantic Detective on one image, writing the final report as JSON.

No local checkpoints, no `ForensicHub`, and no `DTD_TRAIN.yaml` are needed: the
DTD model is built directly from the bundled backbone weights.

### 1. Environment

A CUDA GPU is strongly recommended (the 32B base model is large; expect
multi-GPU or a high-memory GPU). Create an environment and install the runtime
dependencies:

```bash
pip install -e .          # installs the realtext_v2 package + base deps
pip install \
    "torch" \
    "transformers>=4.57" \
    "peft" \
    "huggingface_hub" \
    "accelerate" \
    "paddleocr" "paddlepaddle-gpu" \   # or paddlepaddle (CPU)
    "jpegio" \
    "timm" \
    "efficientnet_pytorch" \
    "opencv-python" "pillow" "numpy"
```

Notes:
- `jpegio` is required for the DTD frequency stream (Y-channel DCT extraction).
- `timm` and `efficientnet_pytorch` are required by the DTD backbones.
- Authenticate to Hugging Face if the base model needs it: `huggingface-cli login`.

### 2. Run

```bash
python scripts/inference/run_two_stage_hf.py \
    --image /path/to/doc.jpg \
    --stage1_prompt prompts/student_prompt_stage1.txt \
    --stage2_prompt prompts/student_prompt_stage2.txt \
    --out report.json \
    --tta \
    --save_dir debug/
```

The first run downloads the HF assets and the base model (cached for reuse).

### 3. Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--image` | — | Input document image (required). |
| `--stage1_prompt` | — | Filterer template (`{{DTD_HINTS}}`). |
| `--stage2_prompt` | — | Detective template (`{{OCR_JSON}}`, `{{FILTERED_DTD}}`). |
| `--out` | `report.json` | Output JSON path. |
| `--dtd_threshold` | `0.40` | DTD probability threshold for regions. |
| `--tta` | off | 4-pass DTD test-time augmentation (min-combine). |
| `--langs` | `en,ch,th,ms,id,ar` | PaddleOCR candidate languages. |
| `--base_model` | `Qwen/Qwen3-VL-32B-Instruct` | Base VLM. |
| `--cache_dir` | HF default | Download cache directory. |
| `--save_dir` | none | Also dump the annotated image and per-stage raw outputs. |

### 4. What it does

1. **DTD** — tiles the image into overlapping 512×512 patches, extracts the
   Y-channel DCT + quant table per patch (`jpegio`), runs DTD, stitches the
   probability map (optionally 4-pass TTA), and extracts numbered regions at the
   threshold.
2. **OCR** — PaddleOCR with language auto-detection and word-level boxes.
3. **Render** — draws the numbered red boxes on the original image.
4. **Stage 1 (Filterer)** — injects all numbered DTD regions into the stage-1
   prompt and decides KEEP/DROP per region.
5. **Stage 2 (Semantic Detective)** — injects only the KEEP regions plus the OCR
   triplets, finds semantic anomalies, and writes the report.
6. Writes `report.json` with the final report and the DTD regions.

### Output

```json
{
  "image_name": "doc.jpg",
  "report": "# FORGERY ANALYSIS REPORT\n...",
  "dtd_threshold": 0.40,
  "n_dtd_regions": 7,
  "dtd_regions": [[x1, y1, x2, y2], ...],
  "ocr_lang": "en"
}
```

## Licence

Code: MIT. Dataset: CC-BY-NC-4.0 (research only), per the RealText-V2 card.
