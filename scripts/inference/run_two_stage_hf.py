#!/usr/bin/env python
"""End-to-end two-stage document-forgery pipeline (single image).

This is a self-contained re-implementation of the third-place GenText-Forensics
pipeline that pulls every required asset from the public Hugging Face dataset
repo and runs the whole chain on ONE image:

    DTD (visual tampering detector)
        -> tampering probability map (tiled 512x512, optional TTA)
        -> numbered candidate regions (threshold 0.40)
    PP-OCRv5 (PaddleOCR)
        -> word-level (text, bbox, conf) triplets + reading-order text
    render the original image with numbered red boxes
    Qwen Filterer        (LoRA)  -> STAGE 1/2: KEEP / DROP each region
    Qwen Semantic Detective (LoRA) -> STAGE 3/4 + final <report>
    -> write the final forensic report as JSON

Assets downloaded from the HF dataset repo
`cmcshnik/GenText-Forensics_third_place_additional_materials`:
    dtd.pth, dtd_qt_table_ori.pk, dtd_backbones/* (model code + backbone weights),
    qwen_filterer/* and qwen_semantic_detective/* (LoRA adapters).
The Qwen3-VL-32B-Instruct base model is fetched from its own HF repo.

The two prompt templates are passed in on the command line; the DTD regions are
injected into stage 1 (all numbered regions) and only the KEEP regions into
stage 2.

Runtime dependencies (must be pip-installed in the environment):
    torch, transformers>=4.57, peft, huggingface_hub, paddleocr, paddlepaddle,
    jpegio, timm, efficientnet_pytorch, opencv-python, pillow, numpy.

Example
-------
    python run_two_stage_hf.py \
        --image doc.jpg \
        --stage1_prompt ../../prompts/student_prompt_stage1.txt \
        --stage2_prompt ../../prompts/student_prompt_stage2.txt \
        --out report.json --tta
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import types
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# OCR helper lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paddle_ocr import run_paddle_ocr_with_lang_detect  # noqa: E402

HF_REPO = "cmcshnik/GenText-Forensics_third_place_additional_materials"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PATCH = 512


# =========================================================================== #
# 1. Asset download
# =========================================================================== #
def download_assets(cache_dir: Optional[str]) -> Path:
    """Download the DTD weights/code and the two LoRA adapters from HF."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        cache_dir=cache_dir,
        allow_patterns=[
            "dtd.pth",
            "dtd_qt_table_ori.pk",
            "dtd_backbones/*.py",
            "dtd_backbones/*.pth",
            "qwen_filterer/*",
            "qwen_semantic_detective/*",
        ],
    )
    return Path(local)


# =========================================================================== #
# 2. DTD model (stub ForensicHub so the bundled model code imports cleanly)
# =========================================================================== #
def _install_forensichub_stubs() -> None:
    """dtd_backbones/dtd.py imports ForensicHub.registry / core.base_model.
    We never need the real package for inference, so inject light stubs."""
    if "ForensicHub" in sys.modules:
        return
    import torch.nn as nn

    fh = types.ModuleType("ForensicHub")
    registry = types.ModuleType("ForensicHub.registry")
    core = types.ModuleType("ForensicHub.core")
    base_model = types.ModuleType("ForensicHub.core.base_model")

    def register_model(name=None):
        def _decorator(cls):
            return cls
        return _decorator

    def register_postfunc(name=None):
        def _decorator(fn):
            return fn
        return _decorator

    registry.register_model = register_model
    registry.register_postfunc = register_postfunc
    base_model.BaseModel = nn.Module

    fh.registry = registry
    fh.core = core
    core.base_model = base_model

    sys.modules["ForensicHub"] = fh
    sys.modules["ForensicHub.registry"] = registry
    sys.modules["ForensicHub.core"] = core
    sys.modules["ForensicHub.core.base_model"] = base_model


def build_dtd(asset_dir: Path, device):
    """Construct the DTD model from the bundled code and load dtd.pth."""
    import torch

    _install_forensichub_stubs()
    sys.path.insert(0, str(asset_dir))  # so `import dtd_backbones.dtd` works
    from dtd_backbones.dtd import DTD  # noqa: E402

    convnext_path = asset_dir / "dtd_backbones" / "convnext_small.pth"
    swin_path = asset_dir / "dtd_backbones" / "swintransformerv2_small.pth"
    model = DTD(convnext_path=str(convnext_path), swin_path=str(swin_path))

    # weights_only=False: the checkpoint bundles training state (argparse.Namespace),
    # which PyTorch >=2.6 blocks under the default weights_only=True. The file comes
    # from our own trusted HF repo.
    try:
        state = torch.load(asset_dir / "dtd.pth", map_location="cpu", weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        state = torch.load(asset_dir / "dtd.pth", map_location="cpu")
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[dtd] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[dtd] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    model.eval().to(device)
    return model


# =========================================================================== #
# 3. DTD inference: DCT extraction, per-patch forward, tiling + TTA
# =========================================================================== #
def _jpeg_dct_qt(patch: Image.Image, quality: int = 95):
    """Return (dct, qt): clip(|Y-DCT|,0,20) int64 [H,W] and the 8x8 luminance
    quantization table, extracted exactly like the original DTD post-function
    (jpegio, first/Y component)."""
    import jpegio

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        patch.convert("RGB").save(tmp.name, "JPEG", quality=quality)
        jpeg = jpegio.read(tmp.name)
    coef = jpeg.coef_arrays[0]                     # Y component DCT coefficients
    qt = jpeg.quant_tables[jpeg.comp_info[0].quant_tbl_no]
    dct = np.clip(np.abs(coef), 0, 20).astype(np.int64)
    return dct, qt.astype(np.int64)


def _dtd_patch_prob(model, patch: Image.Image, device) -> np.ndarray:
    """Run DTD on one 512x512 patch; return the [512,512] tampering probability."""
    import torch

    arr = np.asarray(patch.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    img = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()

    dct_np, qt_np = _jpeg_dct_qt(patch)
    dct = torch.from_numpy(dct_np).unsqueeze(0).unsqueeze(0)        # [1,1,H,W]
    qt = torch.from_numpy(qt_np).unsqueeze(0).unsqueeze(0)          # [1,1,8,8]
    mask = torch.zeros((1, 1, _PATCH, _PATCH), dtype=torch.int64)   # dummy

    img, dct, qt, mask = (t.to(device) for t in (img, dct, qt, mask))
    with torch.inference_mode():
        out = model(img, dct, qt, mask)
    prob = out["pred_mask"][0, 0].float().cpu().numpy()
    return prob


def _tile_starts(size: int, origin: int = 0, patch: int = _PATCH) -> list[int]:
    """Start coordinates for full coverage; border tiles overlap (not cropped)."""
    if size <= patch:
        return [0]
    starts = list(range(origin, size - patch + 1, patch))
    if not starts or starts[0] != 0:
        starts = [0] + starts
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return sorted(set(max(0, min(s, size - patch)) for s in starts))


def _dtd_one_pass(model, image: Image.Image, device, ox: int, oy: int) -> np.ndarray:
    """One full tiling pass at tiling origin (ox, oy); returns a full-size map."""
    W, H = image.size
    padded = image
    pad_w, pad_h = max(W, _PATCH), max(H, _PATCH)
    if (pad_w, pad_h) != (W, H):
        padded = Image.new("RGB", (pad_w, pad_h), (255, 255, 255))
        padded.paste(image, (0, 0))

    prob = np.zeros((pad_h, pad_w), dtype=np.float32)
    count = np.zeros((pad_h, pad_w), dtype=np.float32)
    for y in _tile_starts(pad_h, oy):
        for x in _tile_starts(pad_w, ox):
            patch = padded.crop((x, y, x + _PATCH, y + _PATCH))
            p = _dtd_patch_prob(model, patch, device)
            prob[y:y + _PATCH, x:x + _PATCH] += p
            count[y:y + _PATCH, x:x + _PATCH] += 1.0
    prob /= np.maximum(count, 1.0)
    return prob[:H, :W]


def dtd_infer(model, image: Image.Image, device, tta: bool) -> np.ndarray:
    """Full-image DTD probability map, optionally with 4-pass TTA (min-combine)."""
    if not tta:
        return _dtd_one_pass(model, image, device, 0, 0)
    offsets = [(0, 0), (_PATCH // 2, 0), (0, _PATCH // 2), (_PATCH // 2, _PATCH // 2)]
    maps = [_dtd_one_pass(model, image, device, ox, oy) for ox, oy in offsets]
    return np.minimum.reduce(maps)


# =========================================================================== #
# 4. Region extraction from a probability map (inlined from extract_regions.py)
# =========================================================================== #
def _largest_internal_gap(present: np.ndarray, lo: int, hi: int):
    best_len, best_split, run_start = 0, -1, None
    for i in range(lo, hi + 1):
        if not present[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if run_start > lo and (i - 1) < hi:
                    run_len = i - run_start
                    if run_len > best_len:
                        best_len, best_split = run_len, (run_start + i) // 2
                run_start = None
    return best_len, best_split


def _boxes_from_submask(sub, ox, oy, min_area, min_gap=2, fill_thresh=0.45,
                        depth=0, max_depth=8):
    ys, xs = np.where(sub)
    if ys.size == 0:
        return []
    y1, y2, x1, x2 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h, w = (y2 - y1 + 1), (x2 - x1 + 1)
    area = int(ys.size)
    fill = area / float(max(1, h * w))
    if depth >= max_depth or fill >= fill_thresh or area < 2 * min_area:
        return [[ox + x1, oy + y1, ox + x2 + 1, oy + y2 + 1]] if area >= min_area else []
    rgap_len, rsplit = _largest_internal_gap(sub.any(axis=1), y1, y2)
    cgap_len, csplit = _largest_internal_gap(sub.any(axis=0), x1, x2)
    if max(rgap_len, cgap_len) < min_gap:
        return [[ox + x1, oy + y1, ox + x2 + 1, oy + y2 + 1]] if area >= min_area else []
    out: list = []
    if rgap_len >= cgap_len:
        top = sub.copy(); top[rsplit:, :] = False
        bot = sub.copy(); bot[:rsplit, :] = False
        out += _boxes_from_submask(top, ox, oy, min_area, min_gap, fill_thresh, depth + 1, max_depth)
        out += _boxes_from_submask(bot, ox, oy, min_area, min_gap, fill_thresh, depth + 1, max_depth)
    else:
        left = sub.copy(); left[:, csplit:] = False
        right = sub.copy(); right[:, :csplit] = False
        out += _boxes_from_submask(left, ox, oy, min_area, min_gap, fill_thresh, depth + 1, max_depth)
        out += _boxes_from_submask(right, ox, oy, min_area, min_gap, fill_thresh, depth + 1, max_depth)
    return out


def regions_from_prob(prob: np.ndarray, threshold: float = 0.40, min_area: int = 200,
                      connectivity: int = 4, morph_open_ksize: int = 3) -> list[list[int]]:
    import cv2
    mask = (prob >= threshold).astype(np.uint8)
    if morph_open_ksize and morph_open_ksize >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, labels = cv2.connectedComponents(mask, connectivity=connectivity)
    boxes: list = []
    for lab in range(1, n):
        comp = (labels == lab)
        if int(comp.sum()) < min_area:
            continue
        ys, xs = np.where(comp)
        y1, y2, x1, x2 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        sub = comp[y1:y2 + 1, x1:x2 + 1]
        boxes.extend(_boxes_from_submask(sub, ox=x1, oy=y1, min_area=min_area))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# =========================================================================== #
# 5. Render numbered red boxes on the original image
# =========================================================================== #
def render_red_boxes(image: Image.Image, regions: list[list[int]]) -> Image.Image:
    """Draw numbered red boxes on a copy of the original image. The index label
    height scales with the box height (matching the training-time annotation)."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, (x1, y1, x2, y2) in enumerate(regions, start=1):
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        lw = max(2, (y2 - y1) // 20)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 46, 46), width=lw)
        font_px = max(12, int(0.7 * (y2 - y1)))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_px)
        except Exception:
            font = ImageFont.load_default()
        label = str(i)
        tx, ty = x1, max(0, y1 - font_px - 2)
        try:
            tb = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle(tb, fill=(255, 46, 46))
        except Exception:
            pass
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    return out


# =========================================================================== #
# 6. Prompt assembly (inlined from prerender_prompts.py)
# =========================================================================== #
def _overlap_frac(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ax1, ax2 = sorted((ax1, ax2)); ay1, ay2 = sorted((ay1, ay2))
    bx1, bx2 = sorted((bx1, bx2)); by1, by2 = sorted((by1, by2))
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / max(1, (bx2 - bx1) * (by2 - by1))


def format_dtd_hints(dtd_regions, ocr_items, min_overlap=0.4, max_items=6) -> str:
    if not dtd_regions:
        return "No suspicious regions detected by DTD."
    lines = [f"DTD flagged {len(dtd_regions)} suspicious region(s):"]
    for i, box in enumerate(dtd_regions, start=1):
        x1, y1, x2, y2 = (int(v) for v in box)
        overlaps = []
        for it in ocr_items:
            ob = it.get("bbox") or []
            if len(ob) == 4 and _overlap_frac((x1, y1, x2, y2), ob) >= min_overlap:
                overlaps.append((_overlap_frac((x1, y1, x2, y2), ob), it))
        overlaps.sort(key=lambda kv: -kv[0])
        if overlaps:
            strs = []
            for _, it in overlaps[:max_items]:
                t = (it.get("text") or "").replace('"', "'")
                strs.append(f'#{it.get("id", "?")} "{t[:40]}"')
            info = " | overlaps OCR: " + ", ".join(strs)
        else:
            info = " | no OCR overlap"
        lines.append(f"  Region {i}: [{x1}, {y1}, {x2}, {y2}]{info}")
    return "\n".join(lines)


def format_ocr_compact(ocr_result: dict) -> str:
    items = ocr_result.get("ocr_items", []) or []
    reading = (ocr_result.get("reading_order_text") or "").strip()
    if not items and not reading:
        return "No OCR text detected."
    parts = []
    if items:
        trip = [f'("{it.get("text", "")}", {it.get("bbox", [])}, '
                f'{round(float(it.get("confidence", 0.0)), 3)})' for it in items]
        parts.append("Detected words as (text, [bbox], confidence) triplets:\n\n"
                     + ", ".join(trip))
    if reading:
        parts.append("Reading-order text:\n\n" + reading)
    return "\n\n".join(parts)


_STAGE1_RE = re.compile(r"---\s*STAGE\s*1\s*[:.\-]\s*Knowledge\s*Preparation\s*---", re.I)
_STAGE2_RE = re.compile(r"---\s*STAGE\s*2\s*[:.\-].*?---", re.I)
_STAGE3_RE = re.compile(r"---\s*STAGE\s*3\s*[:.\-].*?---", re.I)
_REGION_BLOCK_RE = re.compile(r"(REGION_\d+\b.*?→\s*(?:KEEP|DROP)\b[^\n.]*\.)",
                              re.DOTALL | re.I)
_KEEP_RE = re.compile(r"→\s*KEEP\b", re.I)
_TAIL_RE = [re.compile(p, re.I) for p in (r"</\s*think\s*>\s*$", r"<\s*think\s*>\s*$",
            r"</\s*tool_call\s*>\s*$", r"```(?:\w+)?\s*$",
            r"\*\*\s*END\s+OF\s+REPORT\s*\*\*\s*$")]


def _trim_tail(body: str) -> str:
    body = body.rstrip()
    for _ in range(20):
        for pat in _TAIL_RE:
            m = pat.search(body)
            if m:
                body = body[: m.start()].rstrip()
                break
        else:
            break
    return body


def build_filtered_dtd(stage1_raw: str) -> str:
    """STAGE 1 text + the KEEP region blocks from STAGE 2 (drops DROP blocks)."""
    m1 = _STAGE1_RE.search(stage1_raw)
    stage1_text = None
    if m1:
        m2 = _STAGE2_RE.search(stage1_raw)
        end = m2.start() if (m2 and m2.start() > m1.start()) else (
            _STAGE3_RE.search(stage1_raw).start()
            if _STAGE3_RE.search(stage1_raw) else len(stage1_raw))
        stage1_text = _trim_tail(stage1_raw[m1.start():end]) or None

    kept = []
    m2 = _STAGE2_RE.search(stage1_raw)
    if m2:
        m3 = _STAGE3_RE.search(stage1_raw)
        seg = stage1_raw[m2.start():(m3.start() if m3 else len(stage1_raw))]
        for m in _REGION_BLOCK_RE.finditer(seg):
            blk = m.group(1).strip()
            if _KEEP_RE.search(blk):
                kept.append(blk)

    pieces = []
    if stage1_text:
        pieces.append(stage1_text)
    pieces.append("--- STAGE 2 KEEP regions (filtered) ---")
    pieces.append("\n\n".join(kept) if kept else "No confirmed tampering regions.")
    if not stage1_text and not kept:
        pieces.insert(0, "(STAGE 1 unavailable)")
    return "\n\n".join(pieces)


def append_image_metadata(prompt: str, w: int, h: int) -> str:
    return prompt + (
        f"\n\nIMAGE METADATA: Width={w} Height={h}. "
        f"All coordinates are absolute integer pixels in this coordinate system."
    )


# =========================================================================== #
# 7. Final report extraction (inlined from infer_student.py)
# =========================================================================== #
_REPORT_ANCHOR_RE = re.compile(r"#\s*FORGERY\s+ANALYSIS\s+REPORT", re.I)
_REPORT_RE = re.compile(r"<report>(.*?)</report>", re.DOTALL | re.I)
_END_MARKER = "**END OF REPORT**"
_STUB_REPORT = (
    "# FORGERY ANALYSIS REPORT\n\n**Overall Assessment:**\n"
    "    **[Conclusion]:** AUTHENTIC\n    **[RISK_SCORE]:** 0\n\n---\n\n"
    "## DETAILED ANOMALY ANALYSIS\n\n(no anomalies detected)\n\n---\n\n"
    "## SUMMARY\nModel failed to produce a schema-compliant report.\n\n**END OF REPORT**\n"
)


def extract_clean_report(text: str) -> str:
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I).strip()
    m = _REPORT_RE.search(text)
    if m:
        text = m.group(1).strip()
    anchors = list(_REPORT_ANCHOR_RE.finditer(text))
    if anchors:
        text = text[anchors[-1].start():]
    end = text.find(_END_MARKER)
    if end >= 0:
        text = text[:end + len(_END_MARKER)]
    text = text.strip()
    return text if _REPORT_ANCHOR_RE.search(text) else _STUB_REPORT


# =========================================================================== #
# 8. Qwen3-VL base + two LoRA adapters
# =========================================================================== #
class TwoAdapterQwen:
    def __init__(self, base_model: str, filterer_dir: Path, detective_dir: Path,
                 device_map="auto", attn_impl="sdpa", max_new_tokens=16384):
        import torch
        import transformers
        from transformers import AutoProcessor
        from peft import PeftModel

        print(f"[qwen] loading base {base_model} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        ModelCls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if ModelCls is None:
            from transformers import AutoModelForVision2Seq as ModelCls  # noqa: N806
        base = ModelCls.from_pretrained(
            base_model, dtype=torch.bfloat16, device_map=device_map,
            attn_implementation=attn_impl, trust_remote_code=True,
        )
        print("[qwen] attaching LoRA adapters (filterer, detective) ...", flush=True)
        self.model = PeftModel.from_pretrained(base, str(filterer_dir),
                                               adapter_name="filterer")
        self.model.load_adapter(str(detective_dir), adapter_name="detective")
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, image: Image.Image, prompt: str, adapter: str) -> str:
        import torch
        self.model.set_adapter(adapter)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=(self.processor.tokenizer.pad_token_id
                          or self.processor.tokenizer.eos_token_id),
        )
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]


# =========================================================================== #
# 9. Main
# =========================================================================== #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Path to a single document image.")
    ap.add_argument("--out", default="report.json", help="Output JSON path.")
    ap.add_argument("--stage1_prompt", required=True,
                    help="Stage-1 (Filterer) prompt template with {{DTD_HINTS}}.")
    ap.add_argument("--stage2_prompt", required=True,
                    help="Stage-2 (Detective) prompt template with {{OCR_JSON}} "
                         "and {{FILTERED_DTD}}.")
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--cache_dir", default=None, help="HF download cache dir.")
    ap.add_argument("--dtd_threshold", type=float, default=0.40)
    ap.add_argument("--min_area", type=int, default=200)
    ap.add_argument("--tta", action="store_true", help="4-pass DTD TTA (min-combine).")
    ap.add_argument("--langs", default="en,ch,th,ms,id,ar",
                    help="Comma-separated PaddleOCR candidate languages.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn_impl", default="sdpa",
                    choices=("eager", "sdpa", "flash_attention_2"))
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--save_dir", default=None,
                    help="Optional dir to also dump intermediate artifacts.")
    return ap.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image_path = Path(args.image).expanduser().resolve()
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    stem = image_path.stem

    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    stage1_template = Path(args.stage1_prompt).read_text(encoding="utf-8")
    stage2_template = Path(args.stage2_prompt).read_text(encoding="utf-8")

    # ---- assets ----
    print("[1/6] downloading assets from HF ...", flush=True)
    asset_dir = download_assets(args.cache_dir)

    # ---- DTD ----
    print("[2/6] DTD inference ...", flush=True)
    dtd_model = build_dtd(asset_dir, device)
    prob = dtd_infer(dtd_model, image, device, tta=args.tta)
    regions = regions_from_prob(prob, threshold=args.dtd_threshold,
                                min_area=args.min_area)
    print(f"      DTD regions @{args.dtd_threshold}: {len(regions)}")

    # ---- OCR ----
    print("[3/6] OCR (PaddleOCR) ...", flush=True)
    ocr = run_paddle_ocr_with_lang_detect(
        image_path, candidate_langs=[s.strip() for s in args.langs.split(",") if s.strip()],
        gpu=(device.type == "cuda"), verbose=False,
    )
    ocr_items = ocr.get("ocr_items", [])

    # ---- annotated image (numbered red boxes) ----
    annotated = render_red_boxes(image, regions)
    if save_dir:
        annotated.save(save_dir / f"{stem}.annotated.png")

    # ---- Qwen (base + 2 LoRA adapters) ----
    print("[4/6] loading Qwen3-VL base + LoRA adapters ...", flush=True)
    qwen = TwoAdapterQwen(
        args.base_model, asset_dir / "qwen_filterer",
        asset_dir / "qwen_semantic_detective",
        attn_impl=args.attn_impl, max_new_tokens=args.max_new_tokens,
    )

    # ---- STAGE 1: Filterer (all numbered regions in the prompt) ----
    print("[5/6] STAGE 1 (Filterer) ...", flush=True)
    stage1_prompt = stage1_template.replace(
        "{{DTD_HINTS}}", format_dtd_hints(regions, ocr_items))
    stage1_prompt = stage1_prompt.replace("{{OCR_JSON}}", "")  # scrub if present
    stage1_prompt = append_image_metadata(stage1_prompt, W, H)
    stage1_raw = qwen.generate(annotated, stage1_prompt, adapter="filterer")
    filtered_dtd = build_filtered_dtd(stage1_raw)
    if save_dir:
        (save_dir / f"{stem}.stage1.txt").write_text(stage1_raw, encoding="utf-8")
        (save_dir / f"{stem}.filtered_dtd.txt").write_text(filtered_dtd, encoding="utf-8")

    # ---- STAGE 2: Semantic Detective (KEEP regions only) ----
    print("[6/6] STAGE 2 (Semantic Detective) ...", flush=True)
    stage2_prompt = (stage2_template
                     .replace("{{OCR_JSON}}", format_ocr_compact(ocr))
                     .replace("{{FILTERED_DTD}}", filtered_dtd)
                     .replace("{{DTD_HINTS}}", ""))
    stage2_prompt = append_image_metadata(stage2_prompt, W, H)
    stage2_raw = qwen.generate(annotated, stage2_prompt, adapter="detective")
    report = extract_clean_report(stage2_raw)
    if save_dir:
        (save_dir / f"{stem}.stage2.txt").write_text(stage2_raw, encoding="utf-8")

    # ---- output ----
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "image_name": image_path.name,
        "report": report,
        "dtd_threshold": args.dtd_threshold,
        "n_dtd_regions": len(regions),
        "dtd_regions": regions,
        "ocr_lang": ocr.get("lang"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
