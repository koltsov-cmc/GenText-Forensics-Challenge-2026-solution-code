#!/usr/bin/env python
"""Sharded multi-GPU inference for trained student VLM with optional vLLM backend.

Parallelism: --num_gpus GPUs per Qwen instance, --num_qwens independent instances.
Inputs are precomputed: images, prompts (.prompt.txt), and optional heatmaps,
organized in flat or partXXX subdirectory layout. Supports HF and vLLM generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = Path(__file__).resolve().parent
_TOOLKIT_ROOT = None
for r in (_SCRIPT_DIR, _SCRIPT_DIR.parent, _SCRIPT_DIR.parent.parent):
    if (r / "realtext_v2").is_dir() or (r / "ForensicHub").is_dir():
        _TOOLKIT_ROOT = r
        break
if _TOOLKIT_ROOT is None:
    _TOOLKIT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_TOOLKIT_ROOT))
sys.path.insert(0, str(_TOOLKIT_ROOT / "scripts"))

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_REPORT_ANCHOR_RE = re.compile(r"#\s*FORGERY\s+ANALYSIS\s+REPORT", re.IGNORECASE)
_END_MARKER = "**END OF REPORT**"


def _rel(part: str, stem: str) -> str:
    """Display / sub-path key. Empty part (flat layout) -> just the stem."""
    return f"{part}/{stem}" if part else stem


# --------------------------------------------------------------------------- #
# Output parsing (verbatim parity with infer_student.py)
# --------------------------------------------------------------------------- #
def _extract_clean_report(text: str) -> str:
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text,
                  flags=re.DOTALL | re.IGNORECASE).strip()
    m = re.search(r"<report>(.*?)</report>", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    anchors = list(_REPORT_ANCHOR_RE.finditer(text))
    if anchors:
        text = text[anchors[-1].start():]
    end_idx = text.find(_END_MARKER)
    if end_idx >= 0:
        text = text[:end_idx + len(_END_MARKER)]
    return text.strip()


def _extract_think_block(text: str) -> Optional[str]:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "STAGE 1" in text.upper() or "STAGE 2" in text.upper():
        return text.strip()
    return None


_STUB_REPORT = """# FORGERY ANALYSIS REPORT

**Overall Assessment:**
    **[Conclusion]:** AUTHENTIC
    **[RISK_SCORE]:** 0

---

## DETAILED ANOMALY ANALYSIS

(no anomalies detected)

---

## SUMMARY
Model failed to produce a schema-compliant report.

**END OF REPORT**
"""


def _has_report_anchor(text: str) -> bool:
    return bool(_REPORT_ANCHOR_RE.search(text))


# --------------------------------------------------------------------------- #
# Smart-resize math (matches Qwen3-VL processor)
# --------------------------------------------------------------------------- #
def _smart_resize_dims(w: int, h: int, max_pixels: int) -> tuple[int, int]:
    if max_pixels <= 0:
        return w, h
    if w * h <= max_pixels:
        return (round(w / 28) * 28, round(h / 28) * 28)
    ratio = math.sqrt(max_pixels / (w * h))
    new_w = max(28, round(w * ratio / 28) * 28)
    new_h = max(28, round(h * ratio / 28) * 28)
    while new_w * new_h > max_pixels and new_w > 28:
        new_w -= 28
    while new_w * new_h > max_pixels and new_h > 28:
        new_h -= 28
    return new_w, new_h


# --------------------------------------------------------------------------- #
# Prompt-length budgeting (truncate text from the END to fit max_model_len)
# --------------------------------------------------------------------------- #
def _estimate_image_tokens(prep: dict, max_pixels: int) -> int:
    """Estimate how many context tokens the image(s) consume after the
    Qwen3-VL smart-resize. Each merged 28x28 patch -> 1 token. Slightly
    over-estimates (no merge-rounding savings) so the budget stays safe."""
    total = 0
    for img in _images_for(prep):
        w, h = img.size
        pw, ph = _smart_resize_dims(w, h, max_pixels)
        total += (max(1, pw // 28)) * (max(1, ph // 28))
    return total


def _truncate_prompt_to_budget(prep: dict, tokenizer, *,
                               max_model_len: int, reserve_output: int,
                               image_tokens: int, margin: int) -> bool:
    """If the text prompt is too long, truncate it FROM THE END so that
    (text_tokens + image_tokens + reserve_output + margin) <= max_model_len.

    Returns True if truncation happened. Mutates prep["prompt"] in place.
    Truncation keeps the BEGINNING of the prompt and drops the tail."""
    text_budget = (max_model_len - reserve_output - image_tokens - margin)
    if text_budget < 64:
        # Degenerate (huge image / tiny window): keep a minimal head.
        text_budget = max(64, max_model_len // 4)

    ids = tokenizer.encode(prep["prompt"], add_special_tokens=False)
    if len(ids) <= text_budget:
        return False
    kept = ids[:text_budget]
    prep["prompt"] = tokenizer.decode(kept, skip_special_tokens=True)
    prep["_truncated_from"] = len(ids)
    prep["_truncated_to"] = text_budget
    return True


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _find_prompt(prompt_part_dir: Path, stem: str) -> Optional[Path]:
    for suffix in (".prompt.txt", ".txt"):
        p = prompt_part_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def _find_heatmap(heatmap_part_dir: Path, stem: str) -> Optional[Path]:
    cands = [
        f"{stem}.heatmap_annotated.png",
        f"{stem}.heatmap.png",
        f"{stem}_heatmap.png",
        f"{stem}_dtd.png",
        f"{stem}.dtd.png",
    ]
    for name in cands:
        p = heatmap_part_dir / name
        if p.exists():
            return p
    return None


def _find_gt_mask(masks_dir: Optional[Path], stem: str) -> Optional[Path]:
    if masks_dir is None or not masks_dir.is_dir():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        for cand in (masks_dir / f"{stem}{ext}",
                     masks_dir / f"{stem}_mask{ext}"):
            if cand.exists():
                return cand
    return None


# --------------------------------------------------------------------------- #
# Work-item discovery
# --------------------------------------------------------------------------- #
def discover_items(args) -> list[dict]:
    image_root   = Path(args.first_input_image).expanduser().resolve()
    prompt_root  = Path(args.prompt_folder).expanduser().resolve()
    heatmap_root = (Path(args.heatmap_dir).expanduser().resolve()
                    if args.heatmap_dir else None)
    needs_heatmap = heatmap_root is not None

    # Flat layout (images directly under image_root) is modelled as a single
    # part with an empty name; "p / ''" is a no-op so all path joins below work
    # unchanged for either layout.
    parts = list(args.parts) if args.parts else [""]

    items: list[dict] = []
    n_missing = {"prompt": 0, "heatmap": 0}

    for part in parts:
        img_part = image_root / part
        prm_part = prompt_root / part
        hm_part  = (heatmap_root / part) if heatmap_root else None

        if not img_part.is_dir():
            print(f"[warn] image dir missing: {img_part}")
            continue
        if not prm_part.is_dir():
            print(f"[warn] prompt dir missing: {prm_part}")
            continue

        for img in sorted(img_part.iterdir()):
            if img.suffix.lower() not in _IMG_EXTS:
                continue
            if args.skip_png and img.suffix.lower() == ".png":
                continue
            stem = img.stem

            prompt_path = _find_prompt(prm_part, stem)
            if prompt_path is None:
                n_missing["prompt"] += 1
                continue

            heatmap_path = None
            if needs_heatmap:
                heatmap_path = _find_heatmap(hm_part, stem)
                if heatmap_path is None:
                    n_missing["heatmap"] += 1
                    continue

            items.append({
                "part":         part,
                "stem":         stem,
                "image_path":   str(img),
                "prompt_path":  str(prompt_path),
                "heatmap_path": str(heatmap_path) if heatmap_path else None,
            })

    layout = "flat" if parts == [""] else f"parts={args.parts}"
    print(f"[discover] {len(items)} item(s)  ({layout})  "
          f"(missing: {n_missing})")
    return items


def _shard(items: list, n_shards: int) -> list[list]:
    """Contiguous near-equal shards."""
    n = len(items)
    base, extra = divmod(n, n_shards)
    out, idx = [], 0
    for k in range(n_shards):
        size = base + (1 if k < extra else 0)
        out.append(items[idx: idx + size]); idx += size
    return out


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #
def _draw_boxes(image, report, out_path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.parse_math"] = False
    import matplotlib.patches as mpatches

    w, h = image.size
    fig_w = min(16, max(6, w / 100))
    fig_h = max(4, fig_w * h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    for a in report.anomalies:
        if not a.grounding or len(a.grounding) != 4:
            continue
        x1, y1, x2, y2 = a.grounding
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        rect = mpatches.Rectangle(
            (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
            linewidth=2, edgecolor="#ff2e2e", facecolor="none")
        ax.add_patch(rect)
        label_bits = [f"#{a.index}"]
        if a.type:
            label_bits.append(a.type[:28])
        ax.text(x1, max(0, y1 - 6), "  ".join(label_bits),
                color="white", fontsize=9,
                bbox=dict(facecolor="#ff2e2e", edgecolor="none",
                           alpha=0.9, pad=2))
    try:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    except Exception:
        try:
            fig.savefig(out_path, dpi=150)
        except Exception:
            pass
    plt.close(fig)


def _save_viz_input_artifacts(viz_dir: Path, part: str, stem: str,
                               image_pil, heatmap_pil, prompt: str,
                               max_image_pixels: int) -> None:
    sample_dir = viz_dir / part / stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    orig_w, orig_h = image_pil.size
    proc_w, proc_h = _smart_resize_dims(orig_w, orig_h, max_image_pixels)
    img = image_pil
    if (proc_w, proc_h) != (orig_w, orig_h):
        img = image_pil.resize((proc_w, proc_h), Image.BILINEAR)
    img.save(sample_dir / f"{stem}_input_image.png")
    if heatmap_pil is not None:
        hm = heatmap_pil
        if hm.size != (proc_w, proc_h):
            hm = hm.resize((proc_w, proc_h), Image.BILINEAR)
        hm.save(sample_dir / f"{stem}_input_heatmap.png")
    (sample_dir / f"{stem}_input_prompt.txt").write_text(prompt, encoding="utf-8")
    (sample_dir / f"{stem}_meta.json").write_text(
        json.dumps({"original_size": [orig_w, orig_h],
                    "processed_size": [proc_w, proc_h],
                    "max_image_pixels": max_image_pixels,
                    "has_heatmap": heatmap_pil is not None},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


def _emit_output_viz(args, image_path: Path, image_pil, report,
                      doc_dir: Path, stem: str) -> None:
    try:
        from vis_report import visualize_report, visualize_report_with_mask
        has_vis = True
    except ImportError:
        has_vis = False

    viz_title = (f"{image_path.name}   |   {report.conclusion}   |   "
                 f"score={report.risk_score}   |   "
                 f"anomalies={len(report.anomalies)}")
    _draw_boxes(image_pil, report, doc_dir / "report_viz.png", title=viz_title)

    if has_vis:
        gt_mask_path = (_find_gt_mask(
            Path(args.gt_masks_dir).expanduser().resolve(), stem)
            if args.gt_masks_dir else None)
        try:
            if gt_mask_path:
                visualize_report_with_mask(
                    image_path, doc_dir / "report.md",
                    doc_dir / "pred_viz.png", gt_mask_path=gt_mask_path,
                    title=f"Pipeline (mask bboxes): {image_path.name}")
            else:
                visualize_report(
                    image_path, doc_dir / "report.md",
                    doc_dir / "pred_viz.png",
                    title=f"Pipeline: {image_path.name}")
        except Exception as exc:
            print(f"  [warn] pred_viz failed: {exc!r}")

        if args.gt_reports:
            gt_base = Path(args.gt_reports).expanduser().resolve()
            gt_path = (gt_base if gt_base.is_file()
                       else gt_base / f"{stem}_report.md")
            if gt_path.exists():
                try:
                    visualize_report(image_path, gt_path,
                                     doc_dir / "gt_viz.png",
                                     title=f"Ground Truth: {image_path.name}")
                except Exception as exc:
                    print(f"  [warn] gt_viz failed: {exc!r}")


# --------------------------------------------------------------------------- #
# Prompt + image preparation
# --------------------------------------------------------------------------- #
def _prepare_item(item: dict, args, tag: str) -> Optional[dict]:
    part = item["part"]; stem = item["stem"]
    image_path = Path(item["image_path"])
    out_doc_dir = Path(args.out_dir) / part / stem   # "p / '' / stem" == "p / stem"

    if args.skip_existing and (out_doc_dir / "report.md").exists():
        return None

    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    prompt = Path(item["prompt_path"]).read_text(encoding="utf-8")

    heatmap_pil = None
    if item["heatmap_path"]:
        heatmap_pil = Image.open(item["heatmap_path"]).convert("RGB")
        if heatmap_pil.size != image_pil.size:
            heatmap_pil = heatmap_pil.resize(image_pil.size, Image.BILINEAR)

    return {
        "part": part, "stem": stem,
        "image_path": image_path, "image_pil": image_pil,
        "heatmap_pil": heatmap_pil, "prompt": prompt,
        "orig_w": orig_w, "orig_h": orig_h, "out_doc_dir": out_doc_dir,
    }


def _persist_outputs(prep: dict, raw_text: str, args, tag: str) -> None:
    doc_dir = prep["out_doc_dir"]
    doc_dir.mkdir(parents=True, exist_ok=True)
    stem = prep["stem"]
    rel = _rel(prep["part"], stem)

    (doc_dir / "prompt.txt").write_text(prep["prompt"], encoding="utf-8")
    if prep["heatmap_pil"] is not None:
        try:
            prep["heatmap_pil"].save(doc_dir / "dtd_overlay.png")
        except Exception:
            pass

    (doc_dir / "report.raw.txt").write_text(raw_text, encoding="utf-8")
    think = _extract_think_block(raw_text)
    if think:
        (doc_dir / "think.txt").write_text(think, encoding="utf-8")

    answer = _extract_clean_report(raw_text)
    if not _has_report_anchor(answer):
        answer = _STUB_REPORT

    from realtext_v2.report import parse_report
    report = parse_report(answer)
    (doc_dir / "report.md").write_text(
        answer + ("\n" if not answer.endswith("\n") else ""), encoding="utf-8")
    (doc_dir / "report.json").write_text(
        json.dumps({"image_name": prep["image_path"].name, "report": answer},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    if args.viz_outputs:
        try:
            _emit_output_viz(args, prep["image_path"], prep["image_pil"],
                             report, doc_dir, stem)
        except Exception as exc:
            print(f"  [{tag}] [warn] output viz failed: {exc!r}")

    print(f"  [{tag}] [{rel}] {report.conclusion}  "
          f"score={report.risk_score}  anomalies={len(report.anomalies)}")


def _messages_for(prep: dict) -> list:
    content = [{"type": "image", "image": prep["image_pil"]}]
    if prep["heatmap_pil"] is not None:
        content.append({"type": "image", "image": prep["heatmap_pil"]})
    content.append({"type": "text", "text": prep["prompt"]})
    return [{"role": "user", "content": content}]


def _images_for(prep: dict) -> list:
    imgs = [prep["image_pil"]]
    if prep["heatmap_pil"] is not None:
        imgs.append(prep["heatmap_pil"])
    return imgs


# --------------------------------------------------------------------------- #
# Backend: HF transformers + PEFT (multi-GPU via device_map, batched gen)
# --------------------------------------------------------------------------- #
class HFBackend:
    def __init__(self, args, max_image_pixels: int, n_gpus: int):
        import torch
        import transformers
        from transformers import AutoProcessor

        self.args = args
        proc_kwargs = {"trust_remote_code": True}
        if max_image_pixels > 0:
            proc_kwargs["max_pixels"] = max_image_pixels
        self.processor = AutoProcessor.from_pretrained(
            args.base_model_id, **proc_kwargs)
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = \
                self.processor.tokenizer.eos_token
        # Left padding is required for correct batched decoder generation.
        self.processor.tokenizer.padding_side = "left"

        ModelCls = getattr(transformers, args.model_class, None)
        if ModelCls is None:
            from transformers import AutoModelForVision2Seq
            ModelCls = AutoModelForVision2Seq

        # Multi-GPU: device_map="auto" shards the model across the visible
        # GPUs (the worker restricted CUDA_VISIBLE_DEVICES to its slice).
        # Single GPU: plain .to("cuda:0").
        device_map = "auto" if n_gpus > 1 else None

        common = dict(dtype=torch.bfloat16,
                      attn_implementation=args.attn_impl,
                      trust_remote_code=True)
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            common["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

        if device_map == "auto":
            model = ModelCls.from_pretrained(
                args.base_model_id, device_map="auto", **common)
        else:
            model = ModelCls.from_pretrained(
                args.base_model_id, device_map=None, **common).to("cuda:0")

        if args.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()
        self.model = model
        # Where to push inputs: first param device works for both.
        self.input_device = next(model.parameters()).device

    def _gen_kwargs(self):
        gk = dict(
            min_new_tokens=self.args.min_new_tokens,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=not self.args.greedy,
            pad_token_id=(self.processor.tokenizer.pad_token_id
                          or self.processor.tokenizer.eos_token_id))
        if not self.args.greedy:
            gk["temperature"] = max(self.args.temperature, 1e-5)
            gk["top_p"] = self.args.top_p
            if self.args.top_k > 0:
                gk["top_k"] = self.args.top_k
        return gk

    def _gen_chunk(self, chunk: list[dict]) -> list[str]:
        import torch
        texts, flat_images = [], []
        for prep in chunk:
            messages = _messages_for(prep)
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            flat_images.extend(_images_for(prep))

        inputs = self.processor(
            text=texts, images=flat_images, padding=True, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {k: (v.to(self.input_device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, **self._gen_kwargs())
        gen = out_ids[:, input_len:]
        return self.processor.batch_decode(
            gen, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

    def generate_batch(self, preps: list[dict], batch_size: int) -> list[str]:
        import torch
        results: list[str] = []
        for i in range(0, len(preps), batch_size):
            chunk = preps[i: i + batch_size]
            try:
                results.extend(self._gen_chunk(chunk))
            except Exception as exc:
                print(f"  [hf] chunk {i}-{i+len(chunk)} failed: {exc!r}; "
                      f"falling back to per-item")
                for prep in chunk:
                    try:
                        results.extend(self._gen_chunk([prep]))
                    except Exception as exc2:
                        print(f"  [hf] item {prep['stem']} failed: {exc2!r}")
                        results.append("")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results


# --------------------------------------------------------------------------- #
# Backend: vLLM (+ LoRA adapter, tensor parallel)
# --------------------------------------------------------------------------- #
class VLLMBackend:
    def __init__(self, args, max_image_pixels: int, n_gpus: int):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoProcessor

        self.args = args
        proc_kwargs = {"trust_remote_code": True}
        if max_image_pixels > 0:
            proc_kwargs["max_pixels"] = max_image_pixels
        self.processor = AutoProcessor.from_pretrained(
            args.base_model_id, **proc_kwargs)

        # primary image (+ optional heatmap) -> up to 2 images per prompt.
        max_imgs = 2 if args.heatmap_dir else 1

        engine_kwargs = dict(
            model=args.base_model_id,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=n_gpus,    # <-- one Qwen across n_gpus cards
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            max_model_len=args.vllm_max_model_len,
            limit_mm_per_prompt={"image": max_imgs},
            mm_processor_kwargs={"max_pixels": max_image_pixels},
            enforce_eager=args.vllm_enforce_eager,
        )
        if args.adapter_path:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_lora_rank"] = args.vllm_max_lora_rank
        self.llm = LLM(**engine_kwargs)

        self.lora_request = (LoRARequest("adapter", 1, args.adapter_path)
                             if args.adapter_path else None)
        if self.lora_request is not None:
            print(f"[vllm] LoRA adapter requested: {args.adapter_path} "
                  f"(max_lora_rank={args.vllm_max_lora_rank})", flush=True)
        elif not args.allow_no_adapter:
            # Should never reach here (parse_args enforces it), but guard
            # anyway so we never silently run base-only.
            raise SystemExit("[vllm] no LoRA adapter and --allow_no_adapter "
                             "not set; refusing to run.")
        self.sampling = SamplingParams(
            temperature=(0.0 if args.greedy else max(args.temperature, 1e-5)),
            top_p=(1.0 if args.greedy else args.top_p),
            top_k=(-1 if (args.greedy or args.top_k <= 0) else args.top_k),
            max_tokens=args.max_new_tokens,
            min_tokens=args.min_new_tokens,
        )

    def _chat_messages(self, prep: dict) -> list:
        content = [{"type": "image_pil", "image_pil": prep["image_pil"]}]
        if prep["heatmap_pil"] is not None:
            content.append({"type": "image_pil",
                            "image_pil": prep["heatmap_pil"]})
        content.append({"type": "text", "text": prep["prompt"]})
        return [{"role": "user", "content": content}]

    def _req(self, prep: dict) -> dict:
        messages = _messages_for(prep)
        prompt_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt_str,
                "multi_modal_data": {"image": _images_for(prep)}}

    def _diag_empty(self, outs, label: str) -> None:
        """Print finish_reason / token counts for the first few empty outputs
        to explain WHY a generation came back empty."""
        shown = 0
        for o in outs:
            comp = o.outputs[0]
            if comp.text.strip():
                continue
            fr = getattr(comp, "finish_reason", "?")
            ntok = len(getattr(comp, "token_ids", []) or [])
            stop = getattr(comp, "stop_reason", None)
            print(f"  [vllm/{label}] EMPTY: finish_reason={fr} "
                  f"n_out_tokens={ntok} stop_reason={stop}", flush=True)
            shown += 1
            if shown >= 3:
                break

    def generate_batch(self, preps: list[dict]) -> list[str]:
        # Preferred path: llm.chat() builds Qwen3-VL multimodal prompts
        # correctly; hand-rolled apply_chat_template can yield empty output.
        if hasattr(self.llm, "chat"):
            try:
                convs = [self._chat_messages(p) for p in preps]
                outs = self.llm.chat(convs, self.sampling,
                                     lora_request=self.lora_request)
                texts = [o.outputs[0].text for o in outs]
                n_empty = sum(1 for t in texts if not t.strip())
                if n_empty:
                    print(f"  [vllm] WARNING {n_empty}/{len(texts)} empty "
                          f"via chat()", flush=True)
                    self._diag_empty(outs, "chat")
                if texts and n_empty < len(texts):
                    return texts
                print("  [vllm] chat() all-empty; trying manual path",
                      flush=True)
            except Exception as exc:
                print(f"  [vllm] chat() failed ({exc!r}); manual path",
                      flush=True)

        reqs = [self._req(p) for p in preps]
        outs = self.llm.generate(reqs, self.sampling,
                                 lora_request=self.lora_request)
        texts = [o.outputs[0].text for o in outs]
        n_empty = sum(1 for t in texts if not t.strip())
        if n_empty:
            print(f"  [vllm] WARNING {n_empty}/{len(texts)} empty "
                  f"via generate()", flush=True)
            self._diag_empty(outs, "generate")
        return texts


# --------------------------------------------------------------------------- #
# Worker (one Qwen, possibly spanning several GPUs)
# --------------------------------------------------------------------------- #
def _run_worker(qwen_id: int, gpu_ids: list[int], shard: list[dict], args,
                return_queue=None):
    # Pin this worker to its slice of GPUs BEFORE importing torch / vllm.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR",
              "MASTER_PORT", "GROUP_RANK"):
        os.environ.pop(k, None)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    tag = f"qwen{qwen_id}"
    n_gpus = len(gpu_ids)
    max_pixels = args.max_image_pixels

    print(f"[{tag}] start  gpus={gpu_ids}  shard={len(shard)}  "
          f"backend={'vllm' if args.vllm_enable else 'hf'}  "
          f"max_pixels={max_pixels}", flush=True)

    t0 = time.time()
    counters = {"ok": 0, "skip": 0, "error": 0}

    viz_inputs_dir = (Path(args.viz_inputs).expanduser().resolve()
                      if args.viz_inputs else None)
    viz_inputs_remaining = (args.viz_inputs_max_samples
                            if viz_inputs_dir else 0)

    try:
        if args.vllm_enable:
            backend = VLLMBackend(args, max_pixels, n_gpus)
        else:
            backend = HFBackend(args, max_pixels, n_gpus)
    except Exception as exc:
        print(f"[{tag}] backend init FAILED: {exc!r}", flush=True)
        import traceback; traceback.print_exc()
        if return_queue is not None:
            return_queue.put({"qwen_id": qwen_id, "counters": counters})
        return counters
    print(f"[{tag}] backend ready in {time.time()-t0:.1f}s", flush=True)

    # Prepare items (load images, fill prompts).
    preps = []
    for item in shard:
        try:
            prep = _prepare_item(item, args, tag)
        except Exception as exc:
            print(f"  [{tag}] prepare error {item['stem']}: {exc!r}")
            counters["error"] += 1
            continue
        if prep is None:
            counters["skip"] += 1
            continue
        preps.append(prep)
        if viz_inputs_dir is not None and viz_inputs_remaining > 0:
            try:
                _save_viz_input_artifacts(
                    viz_inputs_dir, prep["part"], prep["stem"],
                    prep["image_pil"], prep["heatmap_pil"], prep["prompt"],
                    max_pixels)
                viz_inputs_remaining -= 1
            except Exception as exc:
                print(f"  [{tag}] [warn] viz_inputs failed: {exc!r}")

    print(f"[{tag}] prepared {len(preps)} item(s) (skipped {counters['skip']})",
          flush=True)

    # --- Truncate over-long prompts FROM THE END to fit max_model_len ---
    if (not args.no_truncate_prompt) and args.vllm_max_model_len > 0:
        tok = backend.processor.tokenizer
        n_trunc = 0
        worst = 0
        for prep in preps:
            img_tokens = _estimate_image_tokens(prep, max_pixels)
            did = _truncate_prompt_to_budget(
                prep, tok,
                max_model_len=args.vllm_max_model_len,
                reserve_output=args.max_new_tokens,
                image_tokens=img_tokens,
                margin=args.prompt_token_margin)
            if did:
                n_trunc += 1
                worst = max(worst, prep.get("_truncated_from", 0))
        if n_trunc:
            print(f"[{tag}] truncated {n_trunc}/{len(preps)} prompt(s) "
                  f"to fit max_model_len={args.vllm_max_model_len} "
                  f"(reserve_output={args.max_new_tokens}, "
                  f"margin={args.prompt_token_margin}; "
                  f"longest was {worst} text tokens)", flush=True)

    # Generate.
    t_gen = time.time()
    if args.vllm_enable:
        try:
            texts = backend.generate_batch(preps)
        except Exception as exc:
            print(f"[{tag}] vLLM batch FAILED: {exc!r}", flush=True)
            import traceback; traceback.print_exc()
            texts = [""] * len(preps)
    else:
        texts = backend.generate_batch(preps, args.hf_batch_size)

    print(f"[{tag}] generation done in {time.time()-t_gen:.1f}s "
          f"({len(preps)} items)", flush=True)

    # Persist (CPU-bound; done after generation so it doesn't block the GPU).
    for prep, raw in zip(preps, texts):
        try:
            _persist_outputs(prep, raw, args, tag)
            counters["ok"] += 1
        except Exception as exc:
            print(f"  [{tag}] persist error {prep['stem']}: {exc!r}")
            counters["error"] += 1

    print(f"[{tag}] done in {time.time()-t0:.1f}s  {counters}", flush=True)
    if return_queue is not None:
        return_queue.put({"qwen_id": qwen_id, "counters": counters})
    return counters


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Inputs
    ap.add_argument("--first_input_image", required=True,
                    help="Folder with the PRIMARY input images. Either a flat "
                         "folder of images, or a folder of part subdirs "
                         "(part000, part001, ...) selected via --parts.")
    ap.add_argument("--parts", nargs="+", default=None,
                    help="Part subfolders to process (e.g. part000 part001). "
                         "Omit for a FLAT layout (images directly under "
                         "--first_input_image).")
    ap.add_argument("--prompt_folder", required=True)
    ap.add_argument("--heatmap_dir", default=None,
                    help="Optional folder of precomputed heatmaps. When given, "
                         "the heatmap is passed as a second image and items "
                         "without a matching heatmap are skipped.")

    # Model
    ap.add_argument("--base_model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--model_class", default="Qwen3VLForConditionalGeneration")
    ap.add_argument("--attn_impl", default="sdpa",
                    choices=("eager", "sdpa", "flash_attention_2"))
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--allow_no_adapter", action="store_true",
                    help="Permit running WITHOUT a LoRA adapter (bare base "
                         "model). By default this is forbidden and the run "
                         "crashes if --adapter_path is missing/invalid.")
    ap.add_argument("--load_in_4bit", action="store_true")

    # Backend
    ap.add_argument("--vllm_enable", action="store_true")
    ap.add_argument("--vllm_gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--vllm_max_model_len", type=int, default=32768)
    ap.add_argument("--vllm_max_lora_rank", type=int, default=64)
    ap.add_argument("--vllm_enforce_eager", action="store_true")
    ap.add_argument("--hf_batch_size", type=int, default=4,
                    help="(HF backend) items per generate() batch.")
    ap.add_argument("--no_truncate_prompt", action="store_true",
                    help="Disable end-truncation of over-long text prompts. "
                         "By default the prompt is cut from the END so that "
                         "text + image + reserved-output tokens fit within "
                         "--vllm_max_model_len.")
    ap.add_argument("--prompt_token_margin", type=int, default=512,
                    help="Safety margin (tokens) subtracted from the budget "
                         "to account for chat-template wrapper tokens.")

    # Parallelism
    ap.add_argument("--num_gpus", type=int, default=8,
                    help="GPUs per Qwen (tensor/model-parallel width).")
    ap.add_argument("--num_qwens", type=int, default=1,
                    help="Number of independent Qwen instances. Total GPUs "
                         "used = num_gpus * num_qwens.")
    ap.add_argument("--gpu_offset", type=int, default=0,
                    help="First physical GPU id to use (default 0).")
    ap.add_argument("--single_process", action="store_true",
                    help="Run qwen 0 only, in-process (debug).")

    # Resolution
    ap.add_argument("--max_image_pixels", type=int, default=2000000)

    # Generation
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--min_new_tokens", type=int, default=64)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=20)

    # Output
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--skip_png", action="store_true")

    # Viz
    ap.add_argument("--viz_outputs", action="store_true", default=True)
    ap.add_argument("--no_viz_outputs", dest="viz_outputs",
                    action="store_false")
    ap.add_argument("--viz_inputs", default=None)
    ap.add_argument("--viz_inputs_max_samples", type=int, default=40)
    ap.add_argument("--gt_reports", default=None)
    ap.add_argument("--gt_masks_dir", default=None)

    args = ap.parse_args()

    if args.num_gpus < 1 or args.num_qwens < 1:
        raise SystemExit("--num_gpus and --num_qwens must be >= 1")

    # Hard requirement: a trained LoRA adapter MUST be provided and exist.
    # Running on the bare base model by accident would silently produce
    # garbage, so we crash early instead.
    if not args.allow_no_adapter:
        if not args.adapter_path:
            raise SystemExit(
                "ERROR: --adapter_path is required (trained LoRA weights). "
                "Refusing to run on the bare base model. Pass "
                "--allow_no_adapter to override intentionally.")
        ap_dir = Path(args.adapter_path).expanduser()
        cfg = ap_dir / "adapter_config.json"
        wts_ok = any((ap_dir / n).exists() for n in (
            "adapter_model.safetensors", "adapter_model.bin"))
        if not ap_dir.is_dir() or not cfg.exists() or not wts_ok:
            raise SystemExit(
                f"ERROR: --adapter_path '{ap_dir}' is not a valid LoRA "
                f"adapter dir (need adapter_config.json + "
                f"adapter_model.safetensors|bin). Refusing to run.")
    return args


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    items = discover_items(args)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        print("[done] nothing to process")
        return 0

    # GPU assignment: qwen q -> [offset + q*num_gpus : offset + (q+1)*num_gpus]
    n_qwens = args.num_qwens
    if n_qwens > len(items):
        n_qwens = len(items)
    gpu_assignment = []
    for q in range(n_qwens):
        start = args.gpu_offset + q * args.num_gpus
        gpu_assignment.append(list(range(start, start + args.num_gpus)))
    print(f"[plan] {n_qwens} qwen(s), {args.num_gpus} GPU(s) each  "
          f"(total {n_qwens * args.num_gpus} GPUs)")
    for q, gids in enumerate(gpu_assignment):
        print(f"  qwen{q} -> GPUs {gids}")

    shards = _shard(items, n_qwens)
    for q, sh in enumerate(shards):
        print(f"[shard] qwen{q}  size={len(sh)}")

    if args.single_process or n_qwens == 1:
        _run_worker(0, gpu_assignment[0], shards[0], args)
        print(f"\n[done] outputs in {out_dir}/")
        return 0

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    rq = ctx.Queue()
    procs = []
    t_start = time.time()
    for q, (gids, shard) in enumerate(zip(gpu_assignment, shards)):
        if not shard:
            continue
        p = ctx.Process(target=_run_worker, args=(q, gids, shard, args, rq),
                        name=f"qwen{q}")
        p.start()
        time.sleep(5)   # stagger to avoid simultaneous load races
        procs.append(p)
        print(f"[spawn] qwen{q}  pid={p.pid}  gpus={gids}  shard={len(shard)}")

    agg = {"ok": 0, "skip": 0, "error": 0}
    n_done = 0
    while n_done < len(procs):
        try:
            msg = rq.get(timeout=120)
        except Exception:
            if not any(p.is_alive() for p in procs) and rq.empty():
                print("[parent] all workers exited without full report")
                break
            continue
        for k, v in msg["counters"].items():
            agg[k] = agg.get(k, 0) + v
        n_done += 1
        print(f"[parent] qwen{msg['qwen_id']} done {msg['counters']}  "
              f"({n_done}/{len(procs)})")

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            print(f"[parent] terminating stuck {p.name}")
            p.terminate()

    print(f"\n[done] {len(procs)} qwen(s)  elapsed={time.time()-t_start:.1f}s"
          f"  aggregated={agg}\n[done] outputs in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())