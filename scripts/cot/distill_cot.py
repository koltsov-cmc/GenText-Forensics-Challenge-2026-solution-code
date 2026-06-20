#!/usr/bin/env python
"""Distill chain-of-thought from Qwen3-VL-235B-A22B-Instruct teacher model.

For each training image, runs DTD + OCR, reads GT mask/report, builds a teacher
prompt with TP/FP labels, and generates <think>...</think><report>...</report>
output. Supports HF transformers and vLLM backends.

Two modes: CRAFT (live DTD/OCR + prompt construction) and PREBUILT-PROMPT
(precomputed prompts and images keyed by stem).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from transformers import TextStreamer


# --------------------------------------------------------------------------- #
# Path / toolkit setup
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = Path(__file__).resolve().parent
_TOOLKIT_ROOT = None
for r in (_SCRIPT_DIR, _SCRIPT_DIR.parent, _SCRIPT_DIR.parent.parent):
    if (r / "ForensicHub").exists():
        _TOOLKIT_ROOT = r
        break
if _TOOLKIT_ROOT is None:
    _TOOLKIT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_TOOLKIT_ROOT))

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# DTD / OCR / toolkit imports are deferred: in prebuilt-prompt mode we must NOT
# require the DTD or OCR environment at all. They are imported lazily inside
# the craft-mode code path (see _lazy_import_craft_deps).
_dtd = None
_toolkit_mask_to_boxes = None
run_paddle_ocr_subprocess = None


def _lazy_import_craft_deps():
    """Import DTD + OCR + toolkit helpers only when craft mode needs them."""
    global _dtd, _toolkit_mask_to_boxes, run_paddle_ocr_subprocess
    if _dtd is not None:
        return
    _dtd_script_dir = _TOOLKIT_ROOT / "ForensicHub" / "dtd_train"
    sys.path.insert(0, str(_dtd_script_dir))
    sys.path.insert(0, str(_TOOLKIT_ROOT / "scripts"))
    import run_doc_forensics_inference as _dtd_mod
    from run_paddle_subprocess import run_paddle_ocr_subprocess as _ocr_fn
    from realtext_v2.grounding import mask_to_boxes as _mtb
    _dtd = _dtd_mod
    run_paddle_ocr_subprocess = _ocr_fn
    _toolkit_mask_to_boxes = _mtb


# --------------------------------------------------------------------------- #
# Geometry helpers — IoU, bbox extraction from mask, GT report augmentation
# --------------------------------------------------------------------------- #
def _bbox_iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_intersection_over_min(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def mask_to_bboxes(mask_path: Path, min_area: int = 200):
    m = Image.open(str(mask_path)).convert("L")
    arr = np.array(m, dtype=np.uint8)
    binary = (arr > 0).astype(np.uint8)
    if not binary.any():
        return []
    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n_comp):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def classify_dtd_regions(dtd_boxes, mask_bboxes, iou_threshold: float = 0.05):
    out = []
    for d in dtd_boxes:
        is_tp = any(_bbox_intersection_over_min(d, mb) >= iou_threshold
                    for mb in mask_bboxes)
        out.append(is_tp)
    return out


_GROUNDING_RE = re.compile(
    r"\[GROUNDING\]:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)
_CONCLUSION_RE = re.compile(r"\[Conclusion\]:\s*(FORGED|AUTHENTIC)", re.IGNORECASE)


def _extract_conclusion(report_text: str):
    m = _CONCLUSION_RE.search(report_text)
    return m.group(1).upper() if m else None


def augment_gt_report(gt_report_text, mask_bboxes, *, min_overlap: float = 0.10):
    replacements = []

    def _replace(m):
        coords = (int(m.group(1)), int(m.group(2)),
                  int(m.group(3)), int(m.group(4)))
        if not mask_bboxes:
            replacements.append({"original": coords, "replaced_with": None,
                                 "reason": "no mask bboxes"})
            return m.group(0)
        scores = [(_bbox_intersection_over_min(coords, mb), mb)
                  for mb in mask_bboxes]
        scores.sort(reverse=True, key=lambda t: t[0])
        best_score, best_mb = scores[0]
        if best_score >= min_overlap:
            replacements.append({"original": coords, "replaced_with": best_mb,
                                 "score": float(best_score)})
            return f"[GROUNDING]: [{best_mb[0]}, {best_mb[1]}, {best_mb[2]}, {best_mb[3]}]"
        replacements.append({"original": coords, "replaced_with": None,
                             "reason": f"best overlap={best_score:.3f}"})
        return m.group(0)

    augmented = _GROUNDING_RE.sub(_replace, gt_report_text)
    return augmented, replacements


_QWEN_ANOMALY_RE = re.compile(
    r"ANOMALY_(\d+)\s*:.*\n\[GROUNDING\]\s*:\s*\["
    r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
    re.IGNORECASE,
)


def _extract_qwen_coords(raw_output: str):
    coords = {}
    for m in _QWEN_ANOMALY_RE.finditer(raw_output):
        idx = m.group(1)
        coords[idx] = (int(m.group(2)), int(m.group(3)),
                       int(m.group(4)), int(m.group(5)))
    return coords


def _find_anomaly_index(text: str, pos: int):
    search_start = max(0, pos - 2000)
    chunk = text[search_start:pos]
    matches = list(re.finditer(r"###\s*ANOMALY_(\d+)", chunk, re.IGNORECASE))
    if matches:
        return matches[-1].group(1)
    return None


def augment_gt_from_qwen(gt_report_text, raw_output):
    qwen_coords = _extract_qwen_coords(raw_output)
    replacements = []

    def _replace(m):
        coords = (int(m.group(1)), int(m.group(2)),
                  int(m.group(3)), int(m.group(4)))
        idx = _find_anomaly_index(gt_report_text, m.start())
        if idx is not None and idx in qwen_coords:
            qb = qwen_coords[idx]
            replacements.append({"original": coords, "replaced_with": qb,
                                 "source": "qwen"})
            return f"[GROUNDING]: [{qb[0]}, {qb[1]}, {qb[2]}, {qb[3]}]"
        replacements.append({"original": coords, "replaced_with": None,
                             "reason": "no Qwen coord for this anomaly"})
        return m.group(0)

    augmented = _GROUNDING_RE.sub(_replace, gt_report_text)
    return augmented, replacements


def dtd_prob_to_boxes(prob: np.ndarray, threshold: float = 0.4,
                       min_area: int = 200):
    bin_mask = (prob >= threshold).astype(np.uint8) * 255
    raw = _toolkit_mask_to_boxes(bin_mask, min_area=min_area)
    return [tuple(int(v) for v in b) for b in raw]


def format_dtd_hints(dtd_boxes, prob):
    if not dtd_boxes:
        return "No suspicious regions detected by DTD."
    lines = [f"DTD flagged {len(dtd_boxes)} suspicious region(s):"]
    for i, (x1, y1, x2, y2) in enumerate(dtd_boxes, start=1):
        sub = prob[y1:y2, x1:x2]
        conf = float(sub.mean()) if sub.size else 0.0
        lines.append(f"  Region {i}: [{x1}, {y1}, {x2}, {y2}] "
                     f"(mean confidence {conf:.3f})")
    return "\n".join(lines)


def format_tp_fp_labels(dtd_boxes, tp_flags):
    if not dtd_boxes:
        return "(no DTD regions on this image)"
    out = []
    for i, ((x1, y1, x2, y2), is_tp) in enumerate(zip(dtd_boxes, tp_flags),
                                                    start=1):
        tag = "TP (real tampering — KEEP)" if is_tp else "FP (false alarm — DROP)"
        out.append(f"  DTD Region {i} [{x1}, {y1}, {x2}, {y2}]: {tag}")
    return "\n".join(out)


def format_mask_hints(mask_bboxes, dtd_boxes=None, ocr_items=None):
    if not mask_bboxes:
        return "(GT mask is empty — no precise tampering regions)"
    if dtd_boxes is None:
        dtd_boxes = []
    if ocr_items is None:
        ocr_items = []
    out = [f"GT mask contains {len(mask_bboxes)} precise tampering region(s):"]
    for i, (mx1, my1, mx2, my2) in enumerate(mask_bboxes, start=1):
        out.append(f"\n  --- Mask region {i}: [{mx1}, {my1}, {mx2}, {my2}] ---")
        overlapping_dtd = []
        for j, (dx1, dy1, dx2, dy2) in enumerate(dtd_boxes, start=1):
            if _bbox_iou((mx1, my1, mx2, my2), (dx1, dy1, dx2, dy2)) > 0.0:
                overlapping_dtd.append(f"DTD Region {j} [{dx1}, {dy1}, {dx2}, {dy2}]")
        out.append(f"    DTD overlap: {', '.join(overlapping_dtd) if overlapping_dtd else '(none)'}")
        overlapping_ocr = []
        for ocr_item in ocr_items:
            ox1, oy1, ox2, oy2 = ocr_item["bbox"]
            if _bbox_iou((mx1, my1, mx2, my2), (ox1, oy1, ox2, oy2)) > 0.0:
                overlapping_ocr.append(
                    f'#{ocr_item["id"]} "{ocr_item["text"]}" [{ox1}, {oy1}, {ox2}, {oy2}]')
        out.append(f"    OCR overlap: {', '.join(overlapping_ocr) if overlapping_ocr else '(none)'}")
    return "\n".join(out)


def format_ocr_json(ocr_result):
    payload = {
        "reading_order_text": ocr_result["reading_order_text"],
        "selected_language": ocr_result.get("selected_language", "unknown"),
        "language_scores": ocr_result.get("language_scores", {}),
        "ocr_items": [
            {"id": it["id"], "text": it["text"], "bbox": it["bbox"],
             "confidence": round(it["confidence"], 3)}
            for it in ocr_result["ocr_items"]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Teacher output extraction & validation
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_REPORT_RE = re.compile(r"<report>(.*?)</report>", re.DOTALL | re.IGNORECASE)
_STAGE_RE = re.compile(r"---\s*STAGE\s*([1-5])\s*[:.\-]", re.IGNORECASE)
_FORBIDDEN_PHRASES = [
    "given the answer", "since we know", "the gt says",
    "according to the ground truth", "the mask shows",
    "we already know", "as revealed", "it is given that",
    "the ground-truth", "ground truth report",
]


def extract_blocks(text: str):
    think_m = _THINK_RE.search(text)
    rep_m = _REPORT_RE.search(text)
    return ((think_m.group(1).strip() if think_m else None),
            (rep_m.group(1).strip() if rep_m else None))


def validate_teacher_output(raw_text: str, augmented_gt: str) -> dict:
    diag = {"has_think": False, "has_report": False, "stages_found": [],
            "report_match": False, "forbidden_phrase": None, "ok": False}
    think, report = extract_blocks(raw_text)
    if think is None or report is None:
        return diag
    diag["has_think"] = True
    diag["has_report"] = True
    diag["stages_found"] = sorted({int(m.group(1))
                                   for m in _STAGE_RE.finditer(think)})
    norm = lambda s: " ".join(s.split())
    diag["report_match"] = norm(report) == norm(augmented_gt) if augmented_gt else None
    low = think.lower()
    for phrase in _FORBIDDEN_PHRASES:
        if phrase in low:
            diag["forbidden_phrase"] = phrase
            break
    diag["ok"] = (diag["has_think"] and diag["has_report"]
                  and len(diag["stages_found"]) >= 4
                  and (diag["report_match"] in (True, None))
                  and diag["forbidden_phrase"] is None)
    return diag


# --------------------------------------------------------------------------- #
# Teacher backend: HF transformers
# --------------------------------------------------------------------------- #
class TeacherQwen:
    """Lazy-loaded 235B teacher (HF). Generates with 1-2 images + text."""

    def __init__(self, model_id, model_class="Qwen3VLMoeForConditionalGeneration",
                 precision="bf16", max_new_tokens=8192, attn_impl="sdpa"):
        self.model_id = model_id
        self.model_class = model_class
        self.precision = precision
        self.max_new_tokens = max_new_tokens
        self.attn_impl = attn_impl
        self._processor = None
        self._model = None

    def _build_kwargs(self):
        import torch
        from transformers import BitsAndBytesConfig
        kw = {"trust_remote_code": True, "attn_implementation": self.attn_impl}
        if self.precision == "bf16":
            kw["dtype"] = torch.bfloat16; kw["device_map"] = "auto"
        elif self.precision == "int8":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kw["device_map"] = "auto"
        elif self.precision == "int4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kw["device_map"] = "auto"
        else:
            raise ValueError(f"unknown precision: {self.precision}")
        return kw

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import transformers
        from transformers import AutoProcessor
        print(f"[teacher/hf] loading {self.model_id}  precision={self.precision}",
              flush=True)
        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)
        if self._processor.tokenizer.pad_token_id is None:
            self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token
        ModelCls = getattr(transformers, self.model_class, None)
        if ModelCls is None:
            from transformers import AutoModelForVision2Seq
            ModelCls = AutoModelForVision2Seq
            print(f"[teacher] '{self.model_class}' not in transformers; "
                  f"using AutoModelForVision2Seq")
        self._model = ModelCls.from_pretrained(
            self.model_id, **self._build_kwargs()).eval()
        print(f"[teacher/hf] loaded in {time.time()-t0:.1f}s", flush=True)

    def __call__(self, images, prompt_text, use_streamer=True) -> str:
        self._ensure_loaded()
        import torch
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        streamer = TextStreamer(self._processor.tokenizer, skip_prompt=True,
                                skip_special_tokens=True) if use_streamer else None
        gen_kwargs = dict(min_new_tokens=128, max_new_tokens=self.max_new_tokens,
                          max_time=1000.0, do_sample=False,
                          pad_token_id=(self._processor.tokenizer.pad_token_id
                                        or self._processor.tokenizer.eos_token_id))
        if streamer is not None:
            gen_kwargs["streamer"] = streamer
        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)
        input_ids = inputs["input_ids"]
        if (out_ids.shape[1] > input_ids.shape[1]
                and torch.equal(out_ids[:, :input_ids.shape[1]], input_ids)):
            gen_ids = out_ids[:, input_ids.shape[1]:]
        else:
            gen_ids = out_ids
        return self._processor.batch_decode(
            gen_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()

    def generate_batch(self, preps: list[dict]) -> list[str]:
        """HF does not natively batch multimodal well; fall back to loop."""
        return [self(p["images"], p["prompt"], use_streamer=False)
                for p in preps]


# --------------------------------------------------------------------------- #
# Teacher backend: vLLM (235B MoE, tensor_parallel_size=8 ALWAYS)
# --------------------------------------------------------------------------- #
class VLLMTeacher:
    """vLLM teacher. The 235B MoE is sharded across all 8 GPUs (TP=8)."""

    _TP_SIZE = 8   # 235B MoE always spans 8 GPUs

    def __init__(self, model_id, max_new_tokens=8192,
                 gpu_mem_util=0.92, max_model_len=32768,
                 enforce_eager=False, max_image_pixels=0):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.gpu_mem_util = gpu_mem_util
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.max_image_pixels = max_image_pixels
        self._llm = None
        self._sampling = None

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams
        print(f"[teacher/vllm] loading {self.model_id}  TP={self._TP_SIZE}",
              flush=True)
        t0 = time.time()
        engine_kwargs = dict(
            model=self.model_id, trust_remote_code=True, dtype="bfloat16",
            tensor_parallel_size=self._TP_SIZE,
            gpu_memory_utilization=self.gpu_mem_util,
            max_model_len=self.max_model_len,
            limit_mm_per_prompt={"image": 2},
            enforce_eager=self.enforce_eager,
        )
        if self.max_image_pixels > 0:
            engine_kwargs["mm_processor_kwargs"] = {
                "max_pixels": self.max_image_pixels}
        self._llm = LLM(**engine_kwargs)
        self._sampling = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1,
            max_tokens=self.max_new_tokens, min_tokens=128)
        print(f"[teacher/vllm] loaded in {time.time()-t0:.1f}s", flush=True)

    def _truncate_prompt(self, prompt_text: str) -> str:
        """Truncate prompt text so that text tokens + generation budget fit
        within max_model_len. Images are NOT counted — only the text prompt is
        truncated. Keeps the BEGINNING of the prompt (head) because significant
        tokens (system instructions, evidence) are at the start."""
        if self._llm is None:
            return prompt_text
        # Reserve generation tokens + small buffer
        reserve = self.max_new_tokens + 64
        available = self.max_model_len - reserve
        if available <= 0:
            print(f"  [trunc] WARNING max_model_len ({self.max_model_len}) too "
                  f"small for {self.max_new_tokens} gen tokens",
                  flush=True)
            return prompt_text

        tokenizer = self._llm.get_tokenizer()
        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(tokens) <= available:
            return prompt_text

        # Keep the first `available` tokens (head of prompt)
        truncated = tokenizer.decode(tokens[:available],
                                     skip_special_tokens=True)
        print(f"  [trunc] prompt truncated {len(tokens)} -> {available} tokens "
              f"(keep head)", flush=True)
        return truncated

    @staticmethod
    def _chat(images, prompt_text):
        content = [{"type": "image_pil", "image_pil": im} for im in images]
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def __call__(self, images, prompt_text, use_streamer=False) -> str:
        self._ensure_loaded()
        prompt_text = self._truncate_prompt(prompt_text)
        convs = [self._chat(images, prompt_text)]
        outs = self._llm.chat(convs, self._sampling)
        txt = outs[0].outputs[0].text if outs else ""
        if not txt.strip():
            comp = outs[0].outputs[0] if outs else None
            fr = getattr(comp, "finish_reason", "?")
            print(f"  [vllm] WARNING empty output (finish_reason={fr})",
                  flush=True)
        return txt.strip()

    def generate_batch(self, preps: list[dict]) -> list[str]:
        """Batch generation via vLLM continuous batching."""
        self._ensure_loaded()
        for p in preps:
            p["prompt"] = self._truncate_prompt(p["prompt"])
        convs = [self._chat(p["images"], p["prompt"]) for p in preps]
        outs = self._llm.chat(convs, self._sampling)
        texts = [o.outputs[0].text for o in outs]
        n_empty = sum(1 for t in texts if not t.strip())
        if n_empty:
            print(f"  [vllm] WARNING {n_empty}/{len(texts)} empty outputs "
                  f"in batch", flush=True)
        return texts


# --------------------------------------------------------------------------- #
# File discovery (stem-keyed, recursive)
# --------------------------------------------------------------------------- #
def gather_first_images(first_dir: Path, limit: int = 0):
    """Recursively collect primary images. Their stems are the keys used for
    everything else. Supports flat or partXXX/ layouts."""
    paths = sorted(p for p in first_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if limit > 0:
        paths = paths[:limit]
    return paths


def _find_by_stem(folder: Optional[Path], stem: str, exts):
    """Find {folder}/**/{stem}{ext} for the given extensions (recursive)."""
    if folder is None:
        return None
    for ext in exts:
        direct = folder / f"{stem}{ext}"
        if direct.exists():
            return direct
    hits = []
    for ext in exts:
        hits += list(folder.rglob(f"{stem}{ext}"))
    return hits[0] if hits else None


def find_second_image(second_dir: Optional[Path], stem: str):
    return _find_by_stem(second_dir, stem, _IMG_EXTS)


def find_prebuilt_prompt(prompt_folder: Optional[Path], stem: str):
    if prompt_folder is None:
        return None
    for name in (f"{stem}.prompt.txt", f"{stem}.txt", f"{stem}.prompt",
                 f"{stem}.md"):
        direct = prompt_folder / name
        if direct.exists():
            return direct
    for name in (f"{stem}.prompt.txt", f"{stem}.txt"):
        hits = list(prompt_folder.rglob(name))
        if hits:
            return hits[0]
    return None


def find_gt_report(reports_dir: Path, stem: str):
    for suffix in ("_report.md", ".md", "_report.txt", ".txt"):
        p = reports_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def find_gt_mask(masks_dir: Path, stem: str):
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
        for cand in (masks_dir / f"{stem}_mask{ext}",
                     masks_dir / f"{stem}{ext}"):
            if cand.exists():
                return cand
    return None


def _derive_peer_dir(image_dir: Path, peer_name: str) -> Path:
    image_dir = image_dir.resolve()
    parts = list(image_dir.parts)
    for i in reversed(range(len(parts))):
        if parts[i].lower() in ("image", "images", "img"):
            parts[i] = peer_name
            return Path(*parts)
    return image_dir.parent.parent / peer_name / image_dir.name


def _rel_out_subdir(first_image_path: Path, first_root: Path) -> Path:
    """Mirror the partXXX subdir of the primary image under out_dir."""
    try:
        rel = first_image_path.parent.relative_to(first_root)
    except ValueError:
        rel = Path("")
    return rel if rel != Path(".") else Path("")


# --------------------------------------------------------------------------- #
# Visualization (craft-mode logging)
# --------------------------------------------------------------------------- #
def draw_viz(image_pil, dtd_boxes, tp_flags, mask_bboxes, out_path,
             title="", qwen_bboxes=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    w, h = image_pil.size
    fig_w = min(16, max(6, w / 100)); fig_h = max(4, fig_w * h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(image_pil); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    for (x1, y1, x2, y2), is_tp in zip(dtd_boxes, tp_flags):
        x1, x2 = sorted((int(x1), int(x2))); y1, y2 = sorted((int(y1), int(y2)))
        color, label = ("#00ff00", "TP") if is_tp else ("#ff0000", "FP")
        ax.add_patch(mpatches.Rectangle((x1, y1), max(1, x2-x1), max(1, y2-y1),
                     linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.25))
        ax.text(x1, max(0, y1-4), label, color="white", fontsize=7,
                fontweight="bold",
                bbox=dict(facecolor=color, edgecolor="none", alpha=0.85, pad=1.5))
    for i, (x1, y1, x2, y2) in enumerate(mask_bboxes, start=1):
        x1, x2 = sorted((int(x1), int(x2))); y1, y2 = sorted((int(y1), int(y2)))
        ax.add_patch(mpatches.Rectangle((x1, y1), max(1, x2-x1), max(1, y2-y1),
                     linewidth=1.2, edgecolor="#00c853", facecolor="none"))
        ax.text(x2+2, y1+2, f"M{i}", color="#00c853", fontsize=6)
    if qwen_bboxes:
        for i, (x1, y1, x2, y2) in enumerate(qwen_bboxes, start=1):
            x1, x2 = sorted((int(x1), int(x2))); y1, y2 = sorted((int(y1), int(y2)))
            ax.add_patch(mpatches.Rectangle((x1, y1), max(1, x2-x1), max(1, y2-y1),
                         linewidth=1.5, edgecolor="#ff8800", facecolor="none",
                         linestyle="--"))
            ax.text(x2+2, y2+2, f"Q{i}", color="#ff8800", fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Prep / Save split (enables batch generation)
# --------------------------------------------------------------------------- #
def _prep_one(first_image_path, *, args, first_root, out_dir,
              teacher_prompt_template, dtd_model, dtd_model_name,
              dtd_needs_dct, device):
    """Prepare one sample: load images, build/find prompt, return a prep dict.
    Returns {"skipped": True, "stem": stem} if skip_existing hits.
    Returns None on unrecoverable error (missing prompt, missing GT report)."""
    stem = first_image_path.stem
    rel = _rel_out_subdir(first_image_path, first_root)
    doc_out_dir = out_dir / rel
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    out_path_json = doc_out_dir / f"{stem}.cot.json"
    out_path_txt  = doc_out_dir / f"{stem}.cot.txt"
    if args.skip_existing and out_path_json.exists():
        return {"skipped": True, "stem": stem,
                "out_path_json": out_path_json, "out_path_txt": out_path_txt}

    # Load primary + optional second image (generic, stem-keyed).
    image_pil = Image.open(first_image_path).convert("RGB")
    images = [image_pil]
    second_dir = (Path(args.second_input_image).expanduser().resolve()
                  if args.second_input_image else None)
    second_path = find_second_image(second_dir, stem) if second_dir else None
    if second_dir is not None and second_path is None:
        print(f"  [warn] no second image for {stem} in {second_dir}")
    if second_path is not None:
        images.append(Image.open(second_path).convert("RGB"))

    # ============================ PREBUILT-PROMPT MODE ====================== #
    if args.prompt_folder:
        prompt_path = find_prebuilt_prompt(
            Path(args.prompt_folder).expanduser().resolve(), stem)
        if prompt_path is None and args.prompt_folder2:
            prompt_path = find_prebuilt_prompt(
                Path(args.prompt_folder2).expanduser().resolve(), stem)
        if prompt_path is None:
            print(f"  [skip] no prebuilt prompt for {stem}")
            return None
        prompt = prompt_path.read_text(encoding="utf-8")
        return {
            "mode": "prebuilt_prompt",
            "stem": stem,
            "first_image_path": first_image_path,
            "second_path": second_path,
            "prompt_path": prompt_path,
            "images": images,
            "prompt": prompt,
            "out_path_json": out_path_json,
            "out_path_txt": out_path_txt,
        }

    # ============================ CRAFT MODE (legacy) ====================== #
    _lazy_import_craft_deps()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_report_path = find_gt_report(Path(args.gt_reports_dir), stem)
    gt_mask_path   = find_gt_mask(Path(args.gt_masks_dir), stem)
    if gt_report_path is None:
        print(f"  [skip] no GT report for {stem}")
        return None
    gt_report_text = gt_report_path.read_text(encoding="utf-8").strip()
    is_pristine = False
    if gt_mask_path is None:
        report_conclusion = _extract_conclusion(gt_report_text)
        if report_conclusion and report_conclusion.upper() == "FORGED":
            print(f"  [WARNING] mask missing but report says FORGED — skipping {stem}")
            return None
        print(f"  [pristine] no GT mask, report says "
              f"{report_conclusion or 'unknown'} — treating as clean")
        is_pristine = True
        mask_bboxes = []
    else:
        mask_bboxes = mask_to_bboxes(gt_mask_path, min_area=args.mask_min_area)

    print("  [dtd] running ...", flush=True)
    prob, image_pil = _dtd.infer_one_image(
        first_image_path, dtd_model, dtd_model_name, dtd_needs_dct, device,
        jpeg_quality=args.jpeg_quality)
    img_arr = np.asarray(image_pil.convert("RGB"))
    cmap = plt.get_cmap("jet")
    heat = (cmap(prob)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(
        (0.55 * img_arr + 0.45 * heat).clip(0, 255).astype(np.uint8))
    dtd_boxes = dtd_prob_to_boxes(prob, threshold=args.dtd_threshold,
                                   min_area=args.dtd_min_area)
    tp_flags = classify_dtd_regions(dtd_boxes, mask_bboxes,
                                     iou_threshold=args.tp_iou)
    augmented_gt_mask, replacements = augment_gt_report(
        gt_report_text, mask_bboxes, min_overlap=args.coord_replace_overlap)
    n_replaced = sum(1 for r in replacements if r["replaced_with"] is not None)
    print(f"  [coords] {n_replaced}/{len(replacements)} GROUNDINGs -> mask bboxes")

    print("  [ocr] running ...", flush=True)
    ocr_result = run_paddle_ocr_subprocess(
        first_image_path,
        candidate_langs=[s.strip() for s in args.langs.split(",") if s.strip()],
        gpu_id=args.ocr_gpu_id, mag_ratio=1.0)
    ocr_result["selected_language"] = ocr_result["lang"]

    prompt = (teacher_prompt_template
              .replace("{{OCR_JSON}}", format_ocr_json(ocr_result))
              .replace("{{DTD_HINTS}}", format_dtd_hints(dtd_boxes, prob))
              .replace("{{DTD_TP_FP_LABELS}}", format_tp_fp_labels(dtd_boxes, tp_flags))
              .replace("{{MASK_BBOX_HINTS}}",
                       format_mask_hints(mask_bboxes, dtd_boxes, ocr_result["ocr_items"]))
              .replace("{{AUGMENTED_GT_REPORT}}", augmented_gt_mask))

    craft_images = [image_pil, heatmap_pil]
    return {
        "mode": "craft",
        "stem": stem,
        "first_image_path": first_image_path,
        "images": craft_images,
        "prompt": prompt,
        "out_path_json": out_path_json,
        "out_path_txt": out_path_txt,
        "gt_report_text": gt_report_text,
        "augmented_gt_mask": augmented_gt_mask,
        "dtd_boxes": dtd_boxes,
        "tp_flags": tp_flags,
        "mask_bboxes": mask_bboxes,
        "is_pristine": is_pristine,
        "replacements": replacements,
        "ocr_result": ocr_result,
        "prob": prob,
        "image_pil": image_pil,
    }


def _save_one(prep, raw_output, *, args, t_total=None, teacher_elapsed=None):
    """Save CoT output from a prep dict. Returns the record dict."""
    mode = prep["mode"]
    stem = prep["stem"]
    out_path_json = prep["out_path_json"]
    out_path_txt  = prep["out_path_txt"]
    think_block, report_block = extract_blocks(raw_output)

    if mode == "prebuilt_prompt":
        diag = validate_teacher_output(raw_output, "")
        record = {
            "image_name":          prep["first_image_path"].name,
            "stem":                stem,
            "mode":                "prebuilt_prompt",
            "first_image":         str(prep["first_image_path"]),
            "second_image":        str(prep["second_path"]) if prep.get("second_path") else None,
            "prompt_path":         str(prep["prompt_path"]),
            "teacher_model":       args.teacher_model,
            "backend":             "vllm" if args.vllm_enable else "hf",
            "raw_teacher_output":  raw_output,
            "extracted_think":     f"<think>\n{think_block}\n</think>" if think_block else None,
            "extracted_report":    f"<report>\n{report_block}\n</report>" if report_block else None,
            "teacher_elapsed_sec": teacher_elapsed,
            "total_elapsed_sec":   (time.time() - t_total) if t_total else None,
            "validation":          diag,
        }
    else:  # craft mode
        gt_report_text = prep["gt_report_text"]
        augmented_gt_mask = prep["augmented_gt_mask"]
        diag = validate_teacher_output(raw_output, augmented_gt_mask)
        print(f"  [validate] ok={diag['ok']}  stages={diag['stages_found']}  "
              f"report_match={diag['report_match']}  forbidden={diag['forbidden_phrase']}")
        augmented_gt_qwen, qwen_replacements = augment_gt_from_qwen(
            gt_report_text, raw_output)
        record = {
            "image_name": prep["first_image_path"].name, "stem": stem, "mode": "craft",
            "teacher_model": args.teacher_model,
            "backend": "vllm" if args.vllm_enable else "hf",
            "raw_teacher_output": raw_output,
            "extracted_think": f"<think>\n{think_block}\n</think>" if think_block else None,
            "extracted_report": f"<report>\n{report_block}\n</report>" if report_block else None,
            "augmented_gt_report": augmented_gt_qwen,
            "original_gt_report": gt_report_text,
            "dtd_regions": [list(b) for b in prep["dtd_boxes"]],
            "dtd_tp_flags": prep["tp_flags"],
            "mask_bboxes": [list(b) for b in prep["mask_bboxes"]],
            "is_pristine": prep["is_pristine"],
            "coord_replacements": prep["replacements"],
            "qwen_coord_replacements": qwen_replacements,
            "ocr_n_items": prep["ocr_result"]["n_items"],
            "ocr_lang": prep["ocr_result"]["lang"],
            "teacher_elapsed_sec": teacher_elapsed,
            "total_elapsed_sec":   (time.time() - t_total) if t_total else None,
            "validation": diag,
        }
        if args.log and args.log_dir:
            log_dir = Path(args.log_dir) / stem
            log_dir.mkdir(parents=True, exist_ok=True)
            qwen_bboxes = list(_extract_qwen_coords(raw_output).values())
            draw_viz(prep["image_pil"], prep["dtd_boxes"], prep["tp_flags"],
                     prep["mask_bboxes"], log_dir / f"{stem}.viz.png",
                     title=f"Teacher: {stem}", qwen_bboxes=qwen_bboxes)
            print(f"  [log] -> {log_dir}")

    out_path_json.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if think_block and report_block:
        out_path_txt.write_text(
            f"<think>\n{think_block}\n</think>\n\n"
            f"<report>\n{report_block}\n</report>\n", encoding="utf-8")
    print(f"  [saved] {out_path_json.name}")
    return record


def distill_one(first_image_path, *, args, first_root, teacher,
                teacher_prompt_template, out_dir,
                dtd_model=None, dtd_model_name=None, dtd_needs_dct=None,
                device=None):
    """Backward-compatible one-by-one wrapper around _prep_one + _save_one."""
    stem = first_image_path.stem
    t_total = time.time()
    print(f"\n[ {stem} ]")

    prep = _prep_one(
        first_image_path, args=args, first_root=first_root, out_dir=out_dir,
        teacher_prompt_template=teacher_prompt_template,
        dtd_model=dtd_model, dtd_model_name=dtd_model_name,
        dtd_needs_dct=dtd_needs_dct, device=device)

    if prep is None:
        return None
    if prep.get("skipped"):
        return {"stem": stem, "skipped": True}

    print("  [teacher] generating ...", flush=True)
    t0 = time.time()
    raw_output = teacher(prep["images"], prep["prompt"],
                         use_streamer=not args.no_textrender)
    teacher_elapsed = time.time() - t0
    print(f"  [teacher] {teacher_elapsed:.1f}s  out_chars={len(raw_output)}",
          flush=True)

    record = _save_one(prep, raw_output, args=args,
                       t_total=t_total, teacher_elapsed=teacher_elapsed)
    print(f"  [saved] {prep['out_path_json'].name}  "
          f"total={record['total_elapsed_sec']:.1f}s")
    return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Generic image inputs (stem-keyed)
    ap.add_argument("--first_input_image", required=True,
                    help="REQUIRED. Folder of primary images (flat or "
                         "partXXX/). Their stems are the keys used for prompt "
                         "lookup, second-image lookup, and output names.")
    ap.add_argument("--first_input_image2", default=None,
                    help="OPTIONAL. Supplementary primary images. Simply "
                         "concatenated to the list from --first_input_image.")
    ap.add_argument("--second_input_image", default=None,
                    help="OPTIONAL. Folder of a second image per stem (e.g. "
                         "DTD heatmap). Fed alongside the first image. Omit to "
                         "feed a single image.")

    # Prebuilt-prompt mode
    ap.add_argument("--prompt_folder", default=None,
                    help="If set, read the per-image prompt WHOLE from "
                         "{prompt_folder}/{stem}.prompt.txt (already fully "
                         "rendered; NO substitution, NO DTD/OCR/mask). This "
                         "enables prebuilt-prompt mode.")
    ap.add_argument("--prompt_folder2", default=None,
                    help="OPTIONAL. Supplementary prompt folder. If a prompt "
                         "is not found in --prompt_folder, it is looked up "
                         "here. Only meaningful in prebuilt-prompt mode.")

    ap.add_argument("--out_dir", required=True)

    # Craft-mode GT (only used when --prompt_folder is NOT set)
    ap.add_argument("--gt_reports_dir", default=None)
    ap.add_argument("--gt_masks_dir", default=None)

    # DTD (craft mode only)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dtd_threshold", type=float, default=0.4)
    ap.add_argument("--dtd_min_area", type=int, default=200)

    # OCR (craft mode only)
    ap.add_argument("--langs", default="en,ch,th,ms,id,ar")
    ap.add_argument("--ocr_gpu_id", type=int, default=7)

    # Mask handling (craft mode only)
    ap.add_argument("--mask_min_area", type=int, default=100)
    ap.add_argument("--tp_iou", type=float, default=0.05)
    ap.add_argument("--coord_replace_overlap", type=float, default=0.10)

    # Teacher (235B)
    ap.add_argument("--teacher_model", default="Qwen/Qwen3-VL-235B-A22B-Instruct")
    ap.add_argument("--teacher_class", default="Qwen3VLMoeForConditionalGeneration")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "int8", "int4"),
                    help="(HF backend only) bf16/int8/int4.")
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--attn_impl", default="sdpa",
                    choices=("eager", "sdpa", "flash_attention_2"))
    ap.add_argument("--teacher_prompt", default="prompts/teacher_prompt_235b.txt",
                    help="(craft mode) template with {{...}} placeholders.")

    # vLLM backend (235B always TP=8)
    ap.add_argument("--vllm_enable", action="store_true",
                    help="Use vLLM instead of HF. 235B MoE runs with "
                         "tensor_parallel_size=8 (all GPUs).")
    ap.add_argument("--vllm_gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--vllm_max_model_len", type=int, default=15000)
    ap.add_argument("--vllm_enforce_eager", action="store_true")
    ap.add_argument("--vllm_max_image_pixels", type=int, default=0,
                    help="Optional max_pixels for the vLLM mm processor (0=off).")

    # Misc
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--no_textrender", action="store_true")

    args = ap.parse_args()
    # Craft mode requires DTD config/checkpoint.
    if not args.prompt_folder:
        missing = [n for n in ("config", "checkpoint")
                   if getattr(args, n) is None]
        if missing:
            raise SystemExit(
                f"craft mode (no --prompt_folder) requires --{' --'.join(missing)}")
    return args


def main() -> int:
    args = parse_args()

    first_root = Path(args.first_input_image).expanduser().resolve()
    if not first_root.is_dir():
        raise SystemExit(f"--first_input_image not a dir: {first_root}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prebuilt = bool(args.prompt_folder)

    # Craft mode: auto-derive GT dirs + load DTD.
    dtd_model = dtd_model_name = dtd_needs_dct = device = None
    teacher_prompt_template = ""
    if not prebuilt:
        if args.gt_reports_dir is None:
            args.gt_reports_dir = str(_derive_peer_dir(first_root, "report"))
            print(f"[auto] gt_reports_dir -> {args.gt_reports_dir}")
        if args.gt_masks_dir is None:
            args.gt_masks_dir = str(_derive_peer_dir(first_root, "mask"))
            print(f"[auto] gt_masks_dir -> {args.gt_masks_dir}")
        prompt_path = Path(args.teacher_prompt)
        if not prompt_path.is_absolute():
            prompt_path = _TOOLKIT_ROOT / prompt_path
        if not prompt_path.exists():
            alt = _SCRIPT_DIR / args.teacher_prompt
            if alt.exists():
                prompt_path = alt
        if not prompt_path.exists():
            raise SystemExit(f"teacher prompt not found: {args.teacher_prompt}")
        teacher_prompt_template = prompt_path.read_text(encoding="utf-8")
        print(f"[prompt] {prompt_path}  {len(teacher_prompt_template)} chars")

        _lazy_import_craft_deps()
        import torch
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print("[dtd] loading ...")
        t0 = time.time()
        _dtd._setup_paths_and_registry()
        dtd_model, dtd_model_name, dtd_needs_dct = _dtd.build_model_and_load(
            args.config, args.checkpoint, device)
        print(f"[dtd] loaded {dtd_model_name} in {time.time()-t0:.1f}s")
    else:
        print(f"[mode] prebuilt prompts from {args.prompt_folder} "
              f"(no DTD/OCR/mask)")

    if args.log_dir is None and args.log:
        args.log_dir = str(out_dir / "log")
    if args.log and args.log_dir:
        args.log_dir = str(Path(args.log_dir).expanduser().resolve())
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    image_paths = gather_first_images(first_root, limit=args.limit)
    if args.first_input_image2:
        second_root = Path(args.first_input_image2).expanduser().resolve()
        if second_root.is_dir():
            image_paths2 = gather_first_images(second_root, limit=args.limit)
            image_paths = image_paths + image_paths2
            print(f"[run] +{len(image_paths2)} image(s) from {args.first_input_image2}")
        else:
            print(f"[warn] --first_input_image2 not a dir: {second_root}")
    if not image_paths:
        print(f"[error] no images in {first_root}")
        return 1
    print(f"[run] {len(image_paths)} image(s) total")
    print(f"[out] {out_dir}")
    if args.second_input_image:
        print(f"[second] {args.second_input_image}")
    if args.prompt_folder2:
        print(f"[prompt2] {args.prompt_folder2}")

    # Teacher backend.
    if args.vllm_enable:
        teacher = VLLMTeacher(
            model_id=args.teacher_model, max_new_tokens=args.max_new_tokens,
            gpu_mem_util=args.vllm_gpu_mem_util,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,
            max_image_pixels=args.vllm_max_image_pixels)
        print("[backend] vLLM (TP=8)")
    else:
        teacher = TeacherQwen(
            model_id=args.teacher_model, model_class=args.teacher_class,
            precision=args.precision, max_new_tokens=args.max_new_tokens,
            attn_impl=args.attn_impl)
        print("[backend] HF transformers")

    n_done = n_skip = n_fail = n_invalid = 0
    t_start = time.time()

    # ------------------------------------------------------------------ #
    # Batched generation (vLLM only) — prep all, then generate all at once
    # ------------------------------------------------------------------ #
    if args.vllm_enable:
        print("[batch] prep all first ...")
        preps = []
        for i, image_path in enumerate(image_paths, start=1):
            print(f"\n[prep {i}/{len(image_paths)}] {image_path.stem}")
            try:
                prep = _prep_one(
                    image_path, args=args, first_root=first_root, out_dir=out_dir,
                    teacher_prompt_template=teacher_prompt_template,
                    dtd_model=dtd_model, dtd_model_name=dtd_model_name,
                    dtd_needs_dct=dtd_needs_dct, device=device)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {exc!r}")
                import traceback; traceback.print_exc()
                n_fail += 1
                continue
            if prep is None:
                n_fail += 1
            elif prep.get("skipped"):
                n_skip += 1
            else:
                preps.append(prep)

        print(f"[batch] {len(preps)} item(s) ready, generating all at once ...")
        t0 = time.time()
        try:
            outputs = teacher.generate_batch(preps)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] batch generation failed: {exc!r}")
            import traceback; traceback.print_exc()
            n_fail += len(preps)
            outputs = []
        batch_elapsed = time.time() - t0
        per_elapsed = batch_elapsed / max(1, len(preps))
        print(f"  [batch] {batch_elapsed:.1f}s total  "
              f"per_doc≈{per_elapsed:.1f}s")

        for prep, raw_output in zip(preps, outputs):
            try:
                rec = _save_one(prep, raw_output, args=args,
                                teacher_elapsed=per_elapsed)
                n_done += 1
                if not rec["validation"]["ok"]:
                    n_invalid += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] save failed for {prep['stem']}: {exc!r}")
                import traceback; traceback.print_exc()
                n_fail += 1

    # ------------------------------------------------------------------ #
    # One-by-one generation (HF backend)
    # ------------------------------------------------------------------ #
    else:
        for i, image_path in enumerate(image_paths, start=1):
            print(f"\n[{i}/{len(image_paths)}]")
            try:
                rec = distill_one(
                    image_path, args=args, first_root=first_root, teacher=teacher,
                    teacher_prompt_template=teacher_prompt_template, out_dir=out_dir,
                    dtd_model=dtd_model, dtd_model_name=dtd_model_name,
                    dtd_needs_dct=dtd_needs_dct, device=device)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {exc!r}")
                import traceback; traceback.print_exc()
                n_fail += 1
                continue
            if rec is None:
                n_fail += 1
            elif rec.get("skipped"):
                n_skip += 1
            else:
                n_done += 1
                if not rec["validation"]["ok"]:
                    n_invalid += 1

    total = time.time() - t_start
    print(f"\n[done] ok={n_done}  invalid={n_invalid}  "
          f"skipped_existing={n_skip}  failed={n_fail}  total={total/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())