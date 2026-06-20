#!/usr/bin/env python
"""SFT for Qwen3-VL-32B-Instruct on distilled CoT data.

Training modes: full (FSDP full-shard) or lora (QLoRA, 4-bit base + adapter).

Reasoning modes control the target format:
  default:        <think>{full reasoning}</think><report>{augmented_gt}</report>
  --disable_reasoning:  target = <report>...</report> only
  --only_first_stage_reasoning:  target = STAGE 1+2 text
  --only_second_stage_reasoning: target = STAGE 3+4 + report

Uses pre-rendered prompts from prerender_prompts.py. Optional heatmap as second
image via --use_heatmap --heatmap_dir.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

_REASONING_MODES = (
    "default",
    "disable_reasoning",
    "only_first_stage_reasoning",
    "only_second_stage_reasoning",
)


# --------------------------------------------------------------------------- #
# Reasoning extraction
# --------------------------------------------------------------------------- #
_STAGE1_ANCHOR_RE = re.compile(
    r"---\s*STAGE\s*1\s*[:.\-]\s*Knowledge\s*Preparation\s*---", re.IGNORECASE)
_STAGE2_ANCHOR_RE = re.compile(r"---\s*STAGE\s*2\s*[:.\-].*?---", re.IGNORECASE)
_STAGE3_ANCHOR_RE = re.compile(r"---\s*STAGE\s*3\s*[:.\-].*?---", re.IGNORECASE)

_TAIL_GARBAGE_PATTERNS = (
    re.compile(r"</\s*think\s*>\s*$",     re.IGNORECASE),
    re.compile(r"</\s*tool_call\s*>\s*$", re.IGNORECASE),
    re.compile(r"<\s*think\s*>\s*$",      re.IGNORECASE),
    re.compile(r"<\s*tool_call\s*>\s*$",  re.IGNORECASE),
    re.compile(r"```(?:\w+)?\s*$"),
    re.compile(r"\*\*\s*END\s+OF\s+REPORT\s*\*\*\s*$", re.IGNORECASE),
)


def _trim_tail(body: str) -> str:
    """Iteratively strip trailing junk markers."""
    body = body.rstrip()
    for _ in range(20):
        stripped = False
        for pat in _TAIL_GARBAGE_PATTERNS:
            m_tail = pat.search(body)
            if m_tail is not None:
                body = body[: m_tail.start()].rstrip()
                stripped = True
                break
        if not stripped:
            break
    return body


def _extract_reasoning(rec: dict) -> Optional[str]:
    """Extract teacher reasoning from STAGE 1 anchor to end-of-text, with
    iterative trailing-junk trimming. Falls back to extracted_think field if
    raw_teacher_output has no STAGE 1 anchor."""
    candidates: list[str] = []
    raw = rec.get("raw_teacher_output", "") or ""
    if raw:
        candidates.append(raw)
    et = rec.get("extracted_think", "") or ""
    if et:
        candidates.append(et)

    for src in candidates:
        m = _STAGE1_ANCHOR_RE.search(src)
        if m is None:
            continue
        body = _trim_tail(src[m.start():])
        if body:
            return body
    return None


def _extract_first_two_stages(raw_output: str) -> Optional[str]:
    """STAGE 1 anchor up to (excluding) STAGE 3 anchor; tail-trimmed."""
    m1 = _STAGE1_ANCHOR_RE.search(raw_output)
    if not m1:
        return None
    m3 = _STAGE3_ANCHOR_RE.search(raw_output)
    end = m3.start() if m3 else len(raw_output)
    body = _trim_tail(raw_output[m1.start():end])
    return body if body else None


def _extract_stages_3_4(raw_output: str) -> Optional[str]:
    """STAGE 3 anchor to end-of-text; tail-trimmed."""
    m3 = _STAGE3_ANCHOR_RE.search(raw_output)
    if not m3:
        return None
    body = _trim_tail(raw_output[m3.start():])
    return body if body else None


# --------------------------------------------------------------------------- #
# Path resolvers
# --------------------------------------------------------------------------- #
def _load_excluded_stems(exclude_files: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for f in exclude_files:
        if not f.exists():
            print(f"[exclude] WARN: file not found: {f}", flush=True)
            continue
        for ln in f.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            excluded.add(Path(ln).stem)
    return excluded


def _derive_image_path(cot_path: Path, image_root: Path) -> Optional[Path]:
    part = cot_path.parent.name
    stem = (cot_path.name[: -len(".cot.json")]
            if cot_path.name.endswith(".cot.json") else cot_path.stem)
    # 1. partXXX layout (most common)
    part_dir = image_root / part
    if part_dir.is_dir():
        for ext in _IMG_EXTS:
            p = part_dir / f"{stem}{ext}"
            if p.exists():
                return p
    # 2. Flat layout under image_root
    for ext in _IMG_EXTS:
        p = image_root / f"{stem}{ext}"
        if p.exists():
            return p
    # 3. Recursive fallback (image may be in any partXXX subdir)
    for ext in _IMG_EXTS:
        matches = list(image_root.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def _find_heatmap_path(heatmap_dir: Path, part: str,
                        stem: str) -> Optional[Path]:
    """Same lookup priority as before. Includes annotated-heatmap names."""
    candidates = [
        heatmap_dir / part / f"{stem}.heatmap_annotated.png",
        heatmap_dir / part / f"{stem}.heatmap.png",
        heatmap_dir / part / f"{stem}_dtd.png",
        heatmap_dir / part / f"{stem}.dtd.png",
        heatmap_dir / part / f"{stem}_heatmap.png",
        heatmap_dir / f"{stem}.heatmap_annotated.png",
        heatmap_dir / f"{stem}_dtd.png",
        heatmap_dir / f"{stem}.heatmap.png",
        heatmap_dir / f"{stem}.dtd.png",
        heatmap_dir / f"{stem}_heatmap.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_prerendered_prompt(prompts_dir: Path, part: str,
                               stem: str) -> Optional[Path]:
    """Look for {prompts_dir}/{partXXX}/{stem}.prompt.txt."""
    p = prompts_dir / part / f"{stem}.prompt.txt"
    if p.exists():
        return p
    # Fallback: flat layout
    p2 = prompts_dir / f"{stem}.prompt.txt"
    if p2.exists():
        return p2
    return None


def _find_prerendered_target(targets_dir: Path, part: str,
                              stem: str) -> Optional[Path]:
    """Look for {targets_dir}/{partXXX}/{stem}.target.txt."""
    p = targets_dir / part / f"{stem}.target.txt"
    if p.exists():
        return p
    # Fallback: flat layout
    p2 = targets_dir / f"{stem}.target.txt"
    if p2.exists():
        return p2
    return None


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CotJsonlDataset(torch.utils.data.Dataset):
    """Reads .cot.json files under cot_root/partXXX/ and pairs them with
    pre-rendered prompts on disk."""

    def __init__(
        self,
        cot_root: Path,
        image_root: Path,
        prerendered_prompts_dir: Path,
        excluded_stems: set[str],
        require_validation_ok: bool = False,
        use_heatmap: bool = False,
        heatmap_dir: Optional[Path] = None,
        prerendered_targets_dir: Optional[Path] = None,
        *,
        reasoning_mode: str = "default",
        viz_dir: Optional[Path] = None,
        viz_max_samples: int = 50,
        max_image_pixels: int = 0,
        skip_png: bool = False,
    ):
        if reasoning_mode not in _REASONING_MODES:
            raise ValueError(f"reasoning_mode={reasoning_mode!r} not in {_REASONING_MODES}")

        self.image_root              = image_root
        self.prerendered_prompts_dir = prerendered_prompts_dir
        self.prerendered_targets_dir = prerendered_targets_dir
        self.use_heatmap             = use_heatmap
        self.heatmap_dir             = heatmap_dir
        self.reasoning_mode          = reasoning_mode
        self.viz_dir                 = viz_dir
        self.viz_max_samples         = viz_max_samples
        self.max_image_pixels        = max_image_pixels
        self.skip_png                = skip_png
        self.records: list[dict]     = []

        rank = int(os.environ.get("RANK", "0"))

        # Stage-2-reasoning mode MUST run without heatmap (matches stage-2 prompt)
        if self.reasoning_mode == "only_second_stage_reasoning" and self.use_heatmap:
            if rank == 0:
                print("[heatmap] force-OFF for only_second_stage_reasoning",
                      flush=True)
            self.use_heatmap = False

        cot_files: list[Path] = []
        # 1. partXXX subdirs (typical layout)
        for part_dir in sorted(cot_root.iterdir()):
            if part_dir.is_dir() and part_dir.name.lower().startswith("part"):
                cot_files.extend(sorted(part_dir.glob("*.cot.json")))
        # 2. Fallback: flat layout (no partXXX subdirs)
        if not cot_files:
            cot_files = sorted(cot_root.glob("*.cot.json"))
        if rank == 0:
            print(f"[data] scanned {len(cot_files)} cot.json files under {cot_root}",
                  flush=True)

        n_excluded     = 0
        n_no_reason    = 0
        n_no_first_two = 0
        n_no_stages_34 = 0
        n_no_aug_gt    = 0
        n_no_image     = 0
        n_no_heatmap   = 0
        n_no_target    = 0
        n_no_prompt    = 0
        n_bad_valid    = 0
        n_decode_err   = 0
        n_skipped_png  = 0

        for cot_path in cot_files:
            stem = (cot_path.name[: -len(".cot.json")]
                    if cot_path.name.endswith(".cot.json") else cot_path.stem)
            if stem in excluded_stems:
                n_excluded += 1
                continue
            try:
                rec = json.loads(cot_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                n_decode_err += 1
                continue
            if require_validation_ok and not rec.get("validation", {}).get("ok", False):
                n_bad_valid += 1
                continue

            part = cot_path.parent.name

            # ---- Build or load target ----
            if self.prerendered_targets_dir is not None:
                target_path = _find_prerendered_target(
                    self.prerendered_targets_dir, part, stem)
                if target_path is None:
                    n_no_target += 1
                    continue
                target = target_path.read_text(encoding="utf-8")
            else:
                raw_output = rec.get("raw_teacher_output", "") or ""
                augmented_gt = (rec.get("augmented_gt_report")
                                or rec.get("original_gt_report"))
                if self.reasoning_mode == "disable_reasoning":
                    if not augmented_gt:
                        n_no_aug_gt += 1
                        continue
                    target = f"<report>\n{augmented_gt.strip()}\n</report>"
                elif self.reasoning_mode == "only_first_stage_reasoning":
                    first_two = _extract_first_two_stages(raw_output)
                    if not first_two:
                        n_no_first_two += 1
                        continue
                    target = f"<think>\n{first_two.strip()}\n</think>"
                elif self.reasoning_mode == "only_second_stage_reasoning":
                    stages_34 = _extract_stages_3_4(raw_output)
                    if not stages_34:
                        n_no_stages_34 += 1
                        continue
                    if not augmented_gt:
                        n_no_aug_gt += 1
                        continue
                    target = (
                        f"<think>\n{stages_34.strip()}\n</think>\n"
                        f"<report>\n{augmented_gt.strip()}\n</report>"
                    )
                else:  # "default"
                    reasoning = _extract_reasoning(rec)
                    if reasoning is None:
                        n_no_reason += 1
                        continue
                    if not augmented_gt:
                        n_no_aug_gt += 1
                        continue
                    target = (
                        f"<think>\n{reasoning.strip()}\n</think>\n"
                        f"<report>\n{augmented_gt.strip()}\n</report>"
                    )

            # ---- Image ----
            image_path = _derive_image_path(cot_path, image_root)
            if image_path is None:
                n_no_image += 1
                continue

            # ---- skip_png: train only on forged (.jpg) documents ----
            # Convention: .jpg == forged document (has tampering),
            #             .png == clean/pristine document.
            # In skip_png mode we drop every .png so the student only ever
            # sees forged documents and always learns to report anomalies.
            if self.skip_png and image_path.suffix.lower() == ".png":
                n_skipped_png += 1
                continue

            # ---- Heatmap (only if use_heatmap is on) ----
            heatmap_path = None
            if self.use_heatmap and self.heatmap_dir is not None:
                heatmap_path = _find_heatmap_path(self.heatmap_dir, part, stem)
                if heatmap_path is None:
                    n_no_heatmap += 1
                    continue

            # ---- Pre-rendered prompt ----
            prompt_path = _find_prerendered_prompt(
                self.prerendered_prompts_dir, part, stem,
            )
            if prompt_path is None:
                n_no_prompt += 1
                continue

            self.records.append({
                "image_path":   str(image_path),
                "heatmap_path": str(heatmap_path) if heatmap_path else None,
                "prompt_path":  str(prompt_path),
                "target":       target,
                "stem":         stem,
                "is_pristine":  bool(rec.get("is_pristine", False)),
            })

        if rank == 0:
            print(
                f"[data] usable samples: {len(self.records)}  "
                f"mode={self.reasoning_mode}", flush=True,
            )
            print(
                f"[data] skipped: excluded={n_excluded}  "
                f"no_reasoning={n_no_reason}  no_first_two={n_no_first_two}  "
                f"no_stages_34={n_no_stages_34}  no_aug_gt={n_no_aug_gt}  "
                f"no_image={n_no_image}  no_heatmap={n_no_heatmap}  "
                f"no_prompt={n_no_prompt}  no_target={n_no_target}  "
                f"skipped_png={n_skipped_png}  "
                f"validation_bad={n_bad_valid}  decode_err={n_decode_err}",
                flush=True,
            )
            print(
                f"[heatmap] mode={'ON' if self.use_heatmap else 'OFF'}",
                flush=True,
            )

        if self.viz_dir is not None and rank == 0:
            self._save_viz_artifacts()

        if not self.records:
            raise RuntimeError(
                f"no usable cot.json records under {cot_root} "
                f"(prompts_dir={prerendered_prompts_dir}  "
                f"targets_dir={prerendered_targets_dir})"
            )

    @staticmethod
    def _smart_resize(w: int, h: int, max_pixels: int) -> tuple[int, int]:
        """Approximate Qwen3-VL smart_resize."""
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

    def _save_viz_artifacts(self):
        """Save resized image, heatmap, prompt, and target for inspection."""
        from PIL import Image
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for rec in self.records[:self.viz_max_samples]:
            stem = rec["stem"]
            sample_dir = self.viz_dir / stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            image = Image.open(rec["image_path"]).convert("RGB")
            orig_w, orig_h = image.size
            proc_w, proc_h = self._smart_resize(orig_w, orig_h, self.max_image_pixels)
            if (proc_w, proc_h) != (orig_w, orig_h):
                image = image.resize((proc_w, proc_h), Image.BILINEAR)
            image.save(sample_dir / f"{stem}_input_image.png")

            if rec.get("heatmap_path"):
                heatmap = Image.open(rec["heatmap_path"]).convert("RGB")
                if heatmap.size != (proc_w, proc_h):
                    heatmap = heatmap.resize((proc_w, proc_h), Image.BILINEAR)
                heatmap.save(sample_dir / f"{stem}_input_heatmap.png")

            prompt_text = Path(rec["prompt_path"]).read_text(encoding="utf-8")
            (sample_dir / f"{stem}_input_prompt.txt").write_text(
                prompt_text, encoding="utf-8")
            (sample_dir / f"{stem}_target.txt").write_text(
                rec["target"], encoding="utf-8")
            (sample_dir / f"{stem}_meta.json").write_text(json.dumps({
                "original_size":     [orig_w, orig_h],
                "processed_size":    [proc_w, proc_h],
                "max_image_pixels":  self.max_image_pixels,
                "prompt_path":       rec["prompt_path"],
                "heatmap_path":      rec.get("heatmap_path"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            n_saved += 1
        print(f"[viz] saved {n_saved} sample(s) to {self.viz_dir}", flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        from PIL import Image
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        original_size = image.size  # (W, H) before any processing

        # Prompt is loaded from disk per sample
        prompt = Path(rec["prompt_path"]).read_text(encoding="utf-8")

        item = {
            "image":         image,
            "prompt":        prompt,
            "target":        rec["target"],
            "original_size": original_size,
            "stem":          rec["stem"],
        }

        if rec.get("heatmap_path"):
            heatmap = Image.open(rec["heatmap_path"]).convert("RGB")
            # Heatmap MUST match image size before Qwen3-VL processor sees them
            if heatmap.size != image.size:
                heatmap = heatmap.resize(image.size, Image.BILINEAR)
            item["heatmap"] = heatmap

        return item


# --------------------------------------------------------------------------- #
# Collator
# --------------------------------------------------------------------------- #
_BATCH_STATS = {
    "orig_image_sizes":     [],
    "resized_image_grids":  [],
    "n_image_tokens":       [],
    "seq_lengths":          [],
    "seq_length_padded":    0,
    "n_skipped_in_batch":   0,
    "stems":                [],
    # Per-sample truncation status: one of
    #   "none"               — nothing was truncated
    #   "prompt_truncated"   — only the input prompt was shortened from the END
    #   "target_truncated"   — only the target was shortened from the START
    #   "both_truncated"     — prompt shortened AND target shortened
    "truncation_status":    [],
    # Per-sample (used_prompt_chars, used_target_chars) — useful for audits
    "used_prompt_chars":    [],
    "used_target_chars":    [],
    "orig_prompt_chars":    [],
    "orig_target_chars":    [],
}


@dataclass
class MMCollator:
    """Collator for Qwen3-VL multimodal SFT.

    max_seq_length semantics
    ------------------------
    `max_seq_length` is the upper bound on the TOTAL post-tokenisation length
    of (chat-template tokens + image tokens + prompt-text tokens + assistant
    target tokens). When a sample exceeds this bound, the user-message PROMPT
    TEXT is truncated FROM THE END until the full sequence fits.

    Invariants the truncation preserves:
      * The assistant target is always included in full. We never drop or
        truncate the target — it's the supervision signal.
      * Image tokens (including the special <|image_pad|> / image-grid
        scaffolding inserted by the processor) are never split. We truncate
        the prompt STRING before it goes into the processor, so the processor
        always sees a complete prompt and rebuilds image structure cleanly.
      * Truncation removes the trailing characters of the prompt string.
        OCR_JSON / DTD_HINTS / FILTERED_DTD content placed early in the
        prompt is preserved; instruction tail / IMAGE METADATA hint at the
        end of the prompt is what gets shaved.

    If after truncating the prompt down to a near-empty stub the sample STILL
    doesn't fit (i.e. images + target alone exceed max_seq_length), the
    sample is skipped with a counter increment — this is a config problem,
    not a per-sample problem.
    """
    processor: object
    pad_to_multiple_of: int = 8
    max_seq_length: int = 0
    # We always keep at least this many CHARACTERS of the prompt, even when
    # truncating aggressively. Below this we treat the sample as un-trainable
    # (config error) and skip it.
    min_prompt_chars: int = 64
    # We always keep at least this many CHARACTERS of the target. Below this
    # we treat the sample as un-trainable. Target is truncated FROM THE START
    # so the final <report> block is preserved as long as possible.
    min_target_chars: int = 64
    # Max number of (re-tokenise, shrink) iterations per sample before
    # we give up and skip.
    max_truncation_iters: int = 6

    def _tokenize_pair(self, image, heatmap, prompt_text: str, target: str):
        """Run apply_chat_template on (user, user+assistant). Returns
        (user_inputs, full_inputs, prompt_len). Both inputs have token_type_ids
        removed. prompt_len is the position where assistant text begins in
        full_inputs.input_ids."""
        user_content = [{"type": "image", "image": image}]
        if heatmap is not None:
            user_content.append({"type": "image", "image": heatmap})
        user_content.append({"type": "text", "text": prompt_text})

        messages_user = [{"role": "user", "content": user_content}]
        messages_full = messages_user + [{
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        }]

        user_inputs = self.processor.apply_chat_template(
            messages_user, tokenize=True,
            add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        user_inputs.pop("token_type_ids", None)
        prompt_len = user_inputs["input_ids"].shape[1]

        full_inputs = self.processor.apply_chat_template(
            messages_full, tokenize=True,
            add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )
        full_inputs.pop("token_type_ids", None)

        # Realign prompt_len if the two tokenizations diverge before
        # `prompt_len` (chat-template can insert/remove tokens around the
        # assistant boundary).
        ids_full = full_inputs["input_ids"][0]
        ids_user = user_inputs["input_ids"][0]
        common = min(prompt_len, ids_full.shape[0])
        if not torch.equal(ids_full[:common], ids_user[:common]):
            diff = (ids_full[:common] != ids_user[:common]).nonzero(as_tuple=False)
            prompt_len = int(diff[0].item()) if diff.numel() else common

        return user_inputs, full_inputs, prompt_len

    def _fit_sample(self, image, heatmap, prompt_text: str, target: str):
        """Tokenise (image[, heatmap], prompt_text, target) under the
        max_seq_length budget.

        Truncation policy:
          1. First try with full prompt + full target.
          2. If overflow: truncate prompt FROM THE END iteratively, re-tokenise.
             Floor at min_prompt_chars.
          3. If still overflow at floor prompt: truncate target FROM THE START
             iteratively, re-tokenise. Floor at min_target_chars.
             (Truncating target FROM THE START preserves the final <report>
             block in reasoning modes — losing the head of the reasoning is
             far less harmful than losing the report.)
          4. If still overflow at both floors: return None (sample skipped).

        Returns a dict on success:
            {
              "full_inputs":     <processor outputs>,
              "prompt_len":      <int>,
              "used_prompt":     <truncated prompt string>,
              "used_target":     <truncated target string>,
              "status":          one of {"none", "prompt_truncated",
                                          "target_truncated", "both_truncated"},
            }
        Returns None if even with both floors the sample doesn't fit.
        """
        # Pass 1: full sizes
        user_inputs, full_inputs, prompt_len = self._tokenize_pair(
            image, heatmap, prompt_text, target,
        )
        total_len = int(full_inputs["input_ids"].shape[1])

        if not self.max_seq_length or total_len <= self.max_seq_length:
            return {
                "full_inputs": full_inputs,
                "prompt_len":  prompt_len,
                "used_prompt": prompt_text,
                "used_target": target,
                "status":      "none",
            }

        # Pass 2: shrink prompt from the END
        cur_prompt = prompt_text
        cur_target = target
        prompt_was_truncated = False
        prompt_hit_floor = False

        for _ in range(self.max_truncation_iters):
            overflow = total_len - self.max_seq_length
            if overflow <= 0:
                break

            chars_per_token = max(2.0, len(cur_prompt) / max(1, prompt_len))
            chars_to_drop = int(math.ceil(overflow * chars_per_token * 1.10)) + 32
            new_len = len(cur_prompt) - chars_to_drop
            if new_len < self.min_prompt_chars:
                if prompt_hit_floor:
                    break  # already at floor, move on to target truncation
                new_len = self.min_prompt_chars
                prompt_hit_floor = True
            cur_prompt = cur_prompt[:new_len].rstrip()
            prompt_was_truncated = True

            user_inputs, full_inputs, prompt_len = self._tokenize_pair(
                image, heatmap, cur_prompt, cur_target,
            )
            total_len = int(full_inputs["input_ids"].shape[1])
            if total_len <= self.max_seq_length:
                return {
                    "full_inputs": full_inputs,
                    "prompt_len":  prompt_len,
                    "used_prompt": cur_prompt,
                    "used_target": cur_target,
                    "status":      "prompt_truncated",
                }

        # Pass 3: shrink target from the START
        target_was_truncated = False
        target_hit_floor = False

        for _ in range(self.max_truncation_iters):
            overflow = total_len - self.max_seq_length
            if overflow <= 0:
                break

            target_tok_len = total_len - prompt_len
            chars_per_token = max(2.0, len(cur_target) / max(1, target_tok_len))
            chars_to_drop = int(math.ceil(overflow * chars_per_token * 1.10)) + 32
            new_len = len(cur_target) - chars_to_drop
            if new_len < self.min_target_chars:
                if target_hit_floor:
                    return None  # cannot fit even at both floors
                new_len = self.min_target_chars
                target_hit_floor = True
            # Cut FROM THE START — keep the tail (e.g. <report>{...}</report>)
            cur_target = cur_target[-new_len:].lstrip()
            target_was_truncated = True

            user_inputs, full_inputs, prompt_len = self._tokenize_pair(
                image, heatmap, cur_prompt, cur_target,
            )
            total_len = int(full_inputs["input_ids"].shape[1])
            if total_len <= self.max_seq_length:
                status = ("both_truncated" if prompt_was_truncated
                          else "target_truncated")
                return {
                    "full_inputs": full_inputs,
                    "prompt_len":  prompt_len,
                    "used_prompt": cur_prompt,
                    "used_target": cur_target,
                    "status":      status,
                }

        # Last-chance check after the final iteration
        if total_len <= self.max_seq_length:
            if prompt_was_truncated and target_was_truncated:
                status = "both_truncated"
            elif target_was_truncated:
                status = "target_truncated"
            elif prompt_was_truncated:
                status = "prompt_truncated"
            else:
                status = "none"
            return {
                "full_inputs": full_inputs,
                "prompt_len":  prompt_len,
                "used_prompt": cur_prompt,
                "used_target": cur_target,
                "status":      status,
            }
        return None

    def __call__(self, batch: list[dict]):
        input_ids_list, attn_list, labels_list = [], [], []
        pixel_values_list = []
        image_grid_thw_list = []
        mm_token_type_ids_list = []
        n_skipped = 0

        batch_orig_sizes      = []
        batch_grids           = []
        batch_n_imgtokens     = []
        batch_seq_lengths     = []
        batch_stems           = []
        batch_trunc_status    = []
        batch_used_prompt_ch  = []
        batch_used_target_ch  = []
        batch_orig_prompt_ch  = []
        batch_orig_target_ch  = []

        for sample in batch:
            image         = sample["image"]
            heatmap       = sample.get("heatmap")
            prompt        = sample["prompt"]
            target        = sample["target"]
            original_size = sample.get("original_size", (0, 0))
            stem          = sample.get("stem", "?")

            fit = self._fit_sample(image, heatmap, prompt, target)
            if fit is None:
                # Sample can't fit even with both prompt and target shrunk
                # to their floors; skip.
                n_skipped += 1
                continue

            full_inputs  = fit["full_inputs"]
            prompt_len   = fit["prompt_len"]
            used_prompt  = fit["used_prompt"]
            used_target  = fit["used_target"]
            trunc_status = fit["status"]

            ids_full = full_inputs["input_ids"][0]
            attn     = full_inputs["attention_mask"][0]
            labels   = ids_full.clone()
            labels[:prompt_len] = -100

            input_ids_list.append(ids_full)
            attn_list.append(attn)
            labels_list.append(labels)

            if "pixel_values" in full_inputs:
                pixel_values_list.append(full_inputs["pixel_values"])

            grid_thw_tuple = None
            n_img_tokens   = 0
            if "image_grid_thw" in full_inputs:
                image_grid_thw_list.append(full_inputs["image_grid_thw"])
                total_tok = 0
                for row in full_inputs["image_grid_thw"]:
                    t, h, w = row.tolist()
                    total_tok += t * h * w // 4
                n_img_tokens = total_tok
                g0 = full_inputs["image_grid_thw"][0].tolist()
                grid_thw_tuple = tuple(g0)

            if "mm_token_type_ids" in full_inputs:
                mm = full_inputs["mm_token_type_ids"][0]
                mm_token_type_ids_list.append(mm)

            batch_orig_sizes.append(original_size)
            batch_grids.append(grid_thw_tuple)
            batch_n_imgtokens.append(n_img_tokens)
            batch_seq_lengths.append(int(ids_full.shape[0]))
            batch_stems.append(stem)
            batch_trunc_status.append(trunc_status)
            batch_used_prompt_ch.append(len(used_prompt))
            batch_used_target_ch.append(len(used_target))
            batch_orig_prompt_ch.append(len(prompt))
            batch_orig_target_ch.append(len(target))

        if not input_ids_list:
            raise RuntimeError(
                f"all {len(batch)} samples could not fit under "
                f"max_seq_length={self.max_seq_length}. Increase "
                f"--max_seq_length or decrease --max_image_pixels."
            )

        max_len = max(t.shape[0] for t in input_ids_list)
        if self.pad_to_multiple_of:
            r = max_len % self.pad_to_multiple_of
            if r != 0:
                max_len += self.pad_to_multiple_of - r

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        def _pad(t: torch.Tensor, fill: int) -> torch.Tensor:
            if t.shape[0] >= max_len:
                return t[:max_len]
            pad = torch.full((max_len - t.shape[0],), fill, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)

        out = {
            "input_ids":      torch.stack([_pad(t, pad_id) for t in input_ids_list]),
            "attention_mask": torch.stack([_pad(t, 0) for t in attn_list]),
            "labels":         torch.stack([_pad(t, -100) for t in labels_list]),
        }

        if pixel_values_list:
            out["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            out["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
        if mm_token_type_ids_list:
            out["mm_token_type_ids"] = torch.stack(
                [_pad(t, 0) for t in mm_token_type_ids_list])

        _BATCH_STATS["orig_image_sizes"]    = batch_orig_sizes
        _BATCH_STATS["resized_image_grids"] = batch_grids
        _BATCH_STATS["n_image_tokens"]      = batch_n_imgtokens
        _BATCH_STATS["seq_lengths"]         = batch_seq_lengths
        _BATCH_STATS["seq_length_padded"]   = int(out["input_ids"].shape[1])
        _BATCH_STATS["n_skipped_in_batch"]  = n_skipped
        _BATCH_STATS["stems"]               = batch_stems
        _BATCH_STATS["truncation_status"]   = batch_trunc_status
        _BATCH_STATS["used_prompt_chars"]   = batch_used_prompt_ch
        _BATCH_STATS["used_target_chars"]   = batch_used_target_ch
        _BATCH_STATS["orig_prompt_chars"]   = batch_orig_prompt_ch
        _BATCH_STATS["orig_target_chars"]   = batch_orig_target_ch

        return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model_and_processor(args):
    import transformers
    from transformers import AutoProcessor

    rank = int(os.environ.get("RANK", "0"))

    proc_kwargs = {"trust_remote_code": True}
    if args.max_image_pixels > 0:
        proc_kwargs["max_pixels"] = args.max_image_pixels
        if rank == 0:
            est_tokens = args.max_image_pixels // (28 * 28)
            print(f"[proc] max_pixels={args.max_image_pixels}  "
                  f"(~{est_tokens} image tokens max PER IMAGE)", flush=True)
            if args.use_heatmap:
                print(f"[proc] use_heatmap=ON -> 2 images per sample -> "
                      f"~{est_tokens * 2} image tokens total", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ModelCls = getattr(transformers, args.model_class, None)
    if ModelCls is None:
        from transformers import AutoModelForVision2Seq
        ModelCls = AutoModelForVision2Seq
        if rank == 0:
            print(f"[model] '{args.model_class}' not in transformers; "
                  f"using AutoModelForVision2Seq", flush=True)

    if args.train_mode == "full":
        if rank == 0:
            print(f"[model] loading {args.model_id} (bf16, FULL SFT)", flush=True)
        model = ModelCls.from_pretrained(
            args.model_id, dtype=torch.bfloat16,
            attn_implementation=args.attn_impl, trust_remote_code=True,
        )
    elif args.train_mode == "lora":
        from transformers import BitsAndBytesConfig
        if rank == 0:
            print(f"[model] loading {args.model_id} "
                  f"(4-bit base + LoRA r={args.lora_r})", flush=True)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = ModelCls.from_pretrained(
            args.model_id, quantization_config=bnb,
            dtype=torch.bfloat16, attn_implementation=args.attn_impl,
            trust_remote_code=True, device_map={"": local_rank},
        )
    else:
        raise SystemExit(f"unknown train_mode: {args.train_mode}")

    if args.use_liger:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
            apply_liger_kernel_to_qwen3_vl(
                rope=True, fused_linear_cross_entropy=True,
                rms_norm=True, swiglu=True, model=model,
            )
            if rank == 0:
                print("[liger] patched", flush=True)
        except ImportError:
            if rank == 0:
                print("[liger] not installed; skipping", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"[liger] patch failed: {e}", flush=True)

    n_frozen, n_train = 0, 0
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in ("visual", "vision",
                                            "patch_embed", "image_encoder")):
            p.requires_grad_(False)
            n_frozen += p.numel()
        else:
            n_train += p.numel()
    if rank == 0:
        print(f"[model] params before LoRA: trainable={n_train/1e9:.2f}B  "
              f"frozen={n_frozen/1e9:.2f}B", flush=True)

    if args.train_mode == "lora":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False,
        )
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")
                          if m.strip()]
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            task_type="CAUSAL_LM", target_modules=target_modules,
        )
        model = get_peft_model(model, peft_cfg)

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "base_model") and hasattr(
                model.base_model, "enable_input_require_grads"
            ):
                model.base_model.enable_input_require_grads()

        if rank == 0:
            model.print_trainable_parameters()
            print("[model] gradient checkpointing enabled "
                  "(use_reentrant=False) with input_require_grads", flush=True)
    else:
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if rank == 0:
                print("[model] gradient checkpointing enabled "
                      "(use_reentrant=False)", flush=True)

    return model, processor


# --------------------------------------------------------------------------- #
# Per-step stats callback
# --------------------------------------------------------------------------- #
def _make_stats_callback_class():
    from transformers import TrainerCallback

    class StepStatsCallback(TrainerCallback):
        def __init__(self, output_dir: Path, print_every_n_steps: int = 1):
            import time
            super().__init__()
            self.output_dir = Path(output_dir)
            self.print_every = print_every_n_steps
            self._last_step_time = time.time()
            self._step_count_global = 0
            self._jsonl_fh = None
            self._rank = int(os.environ.get("RANK", "0"))
            self._latest_loss = None
            self._latest_lr   = None

        def _open_jsonl(self):
            if self._rank == 0 and self._jsonl_fh is None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._jsonl_fh = open(self.output_dir / "step_stats.jsonl",
                                       "a", buffering=1)

        def on_train_begin(self, args, state, control, **kwargs):
            import time
            self._open_jsonl()
            self._last_step_time = time.time()
            self._latest_loss = None
            self._latest_lr   = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None: return
            if "loss" in logs: self._latest_loss = logs["loss"]
            if "learning_rate" in logs: self._latest_lr = logs["learning_rate"]

        def on_step_end(self, args, state, control, **kwargs):
            import time
            now = time.time()
            step_time = now - self._last_step_time
            self._last_step_time = now
            self._step_count_global = state.global_step

            stats = dict(_BATCH_STATS)
            loss = self._latest_loss
            lr   = self._latest_lr

            if self._rank != 0:
                return

            seq_lengths   = stats.get("seq_lengths", [])
            n_img_tokens  = stats.get("n_image_tokens", [])
            orig_sizes    = stats.get("orig_image_sizes", [])
            grids         = stats.get("resized_image_grids", [])
            padded_len    = stats.get("seq_length_padded", 0)
            n_skipped     = stats.get("n_skipped_in_batch", 0)
            stems         = stats.get("stems", [])
            trunc_status  = stats.get("truncation_status", [])
            used_p_ch     = stats.get("used_prompt_chars", [])
            used_t_ch     = stats.get("used_target_chars", [])
            orig_p_ch     = stats.get("orig_prompt_chars", [])
            orig_t_ch     = stats.get("orig_target_chars", [])

            compressed_resolutions = []
            for g in grids:
                if g is None:
                    compressed_resolutions.append(None)
                else:
                    compressed_resolutions.append((g[2] * 14, g[1] * 14))

            # Truncation counters for compact display
            n_prompt_trunc = sum(1 for s in trunc_status if s in ("prompt_truncated", "both_truncated"))
            n_target_trunc = sum(1 for s in trunc_status if s in ("target_truncated", "both_truncated"))
            n_both         = sum(1 for s in trunc_status if s == "both_truncated")

            record = {
                "step":           state.global_step,
                "epoch":          state.epoch,
                "loss":           loss,
                "lr":             lr,
                "step_time_sec":  round(step_time, 3),
                "seq_length_padded":         padded_len,
                "seq_lengths_per_sample":    seq_lengths,
                "n_image_tokens_per_sample": n_img_tokens,
                "orig_image_sizes_wh":       [list(s) for s in orig_sizes],
                "compressed_image_sizes_wh": [list(s) if s else None
                                              for s in compressed_resolutions],
                "n_skipped_in_batch":        n_skipped,
                "stems":                     stems,
                # Per-sample truncation info — one entry per sample in the
                # batch, aligned with `stems`. Lets you grep the JSONL for
                # which samples got cropped and how badly.
                "truncation_status":         trunc_status,
                "used_prompt_chars":         used_p_ch,
                "used_target_chars":         used_t_ch,
                "orig_prompt_chars":         orig_p_ch,
                "orig_target_chars":         orig_t_ch,
                "n_prompt_truncated":        n_prompt_trunc,
                "n_target_truncated":        n_target_trunc,
                "n_both_truncated":          n_both,
            }
            if self._jsonl_fh is not None:
                self._jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            if state.global_step % self.print_every == 0:
                loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "?"
                lr_str   = f"{lr:.2e}" if isinstance(lr, (int, float)) else "?"
                seq_str  = (f"max={max(seq_lengths)}/avg={sum(seq_lengths)//max(1,len(seq_lengths))}"
                            if seq_lengths else "?")
                img_tok_str = (f"avg={sum(n_img_tokens)//max(1,len(n_img_tokens))}"
                               if n_img_tokens else "?")
                orig_str = (f"{orig_sizes[0][0]}x{orig_sizes[0][1]}"
                            if orig_sizes else "?")
                comp_str = (f"{compressed_resolutions[0][0]}x{compressed_resolutions[0][1]}"
                            if compressed_resolutions and compressed_resolutions[0] else "?")
                skip_str  = f" skip={n_skipped}" if n_skipped > 0 else ""
                trunc_parts = []
                if n_prompt_trunc:
                    trunc_parts.append(f"prompt={n_prompt_trunc}")
                if n_target_trunc:
                    trunc_parts.append(f"target={n_target_trunc}")
                if n_both:
                    trunc_parts.append(f"both={n_both}")
                trunc_str = f" trunc[{','.join(trunc_parts)}]" if trunc_parts else ""
                print(
                    f"[step {state.global_step:>5d}] "
                    f"loss={loss_str} lr={lr_str} t={step_time:5.2f}s "
                    f"seq[{seq_str}|pad={padded_len}] "
                    f"img_tok[{img_tok_str}] img[{orig_str} -> {comp_str}]"
                    f"{skip_str}{trunc_str}",
                    flush=True,
                )

        def on_train_end(self, args, state, control, **kwargs):
            if self._jsonl_fh is not None:
                self._jsonl_fh.close()
                self._jsonl_fh = None

    return StepStatsCallback


def _make_epoch_checkpoint_callback_class():
    """Factory for an HF TrainerCallback that preserves epoch-boundary
    checkpoints from save_total_limit rotation.

    HF Trainer's save_total_limit rotates checkpoints in a rolling window,
    deleting older ones. That's fine for resume-from-crash, but if you want
    to compare epochs or roll back to "the model after epoch 2", you'll lose
    those checkpoints to rotation.

    This callback:
      1. on_epoch_end -> set control.should_save = True so Trainer writes
         a checkpoint at the next step (or immediately, depending on
         Trainer's flow). Also stores the epoch index for the next save.
      2. on_save -> after Trainer finishes the save, COPY the resulting
         `checkpoint-{step}` directory into
         `{output_dir}/epoch_checkpoints/epoch_{N:03d}_step_{S}/`. This
         copy is outside Trainer's rotation tracking and won't be deleted
         by save_total_limit.

    Only rank 0 performs the copy (filesystem op, not collective).

    Disk-space note: each preserved epoch is a full duplicate of the LoRA
    adapter + trainer state. For LoRA r=16 that's ~50-100 MB per epoch
    (negligible). For FSDP full-shard checkpoints it can be GBs per epoch —
    set --num_train_epochs sensibly or prune manually.
    """
    import shutil
    from transformers import TrainerCallback

    class EpochCheckpointCallback(TrainerCallback):
        def __init__(self, output_dir: Path):
            super().__init__()
            self.output_dir   = Path(output_dir)
            self.epoch_dir    = self.output_dir / "epoch_checkpoints"
            self._rank        = int(os.environ.get("RANK", "0"))
            # Index of the epoch boundary that just ended; consumed by next
            # on_save. None = no epoch save pending.
            self._pending_epoch: Optional[int] = None

        def on_epoch_end(self, args, state, control, **kwargs):
            # Round epoch to nearest int — at on_epoch_end state.epoch is
            # typically e.g. 1.0, 2.0, ... (clean boundary).
            self._pending_epoch = int(round(state.epoch)) if state.epoch else 0
            # Ask Trainer to save now. Trainer will call on_save() after
            # writing checkpoint-{global_step}.
            control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            # Only act if we requested this save for an epoch boundary.
            # (Regular step-based saves leave _pending_epoch as None and
            # are handled normally by save_total_limit rotation.)
            if self._pending_epoch is None:
                return control
            if self._rank != 0:
                # Other ranks: clear flag, don't touch filesystem.
                self._pending_epoch = None
                return control

            step = state.global_step
            ckpt_src = self.output_dir / f"checkpoint-{step}"
            if not ckpt_src.is_dir():
                print(
                    f"[epoch_ckpt] WARN: expected {ckpt_src} not found at "
                    f"on_save; epoch {self._pending_epoch} not preserved",
                    flush=True,
                )
                self._pending_epoch = None
                return control

            self.epoch_dir.mkdir(parents=True, exist_ok=True)
            dst = (self.epoch_dir
                   / f"epoch_{self._pending_epoch:03d}_step_{step}")
            if dst.exists():
                print(
                    f"[epoch_ckpt] {dst.name} already exists, skipping copy",
                    flush=True,
                )
            else:
                try:
                    shutil.copytree(ckpt_src, dst)
                    sz_mb = sum(p.stat().st_size for p in dst.rglob("*")
                                if p.is_file()) / 1e6
                    print(
                        f"[epoch_ckpt] preserved {dst.name}  "
                        f"({sz_mb:.0f} MB)",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[epoch_ckpt] ERROR copying {ckpt_src.name} -> "
                        f"{dst.name}: {exc!r}",
                        flush=True,
                    )

            self._pending_epoch = None
            return control

    return EpochCheckpointCallback


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--train_mode", default="full", choices=("full", "lora"))

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules",
                    default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--merge_after_training", action="store_true")

    ap.add_argument("--cot_root",   required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument(
        "--prerendered_prompts_dir", required=True,
        help="Directory of pre-rendered per-sample prompts. Layout: "
             "{this_dir}/{partXXX}/{stem}.prompt.txt. Generate with "
             "prerender_prompts.py.",
    )
    ap.add_argument(
        "--prerendered_targets_dir", default=None,
        help="Directory of pre-rendered per-sample targets. Layout: "
             "{this_dir}/{partXXX}/{stem}.target.txt. When set, targets are "
             "loaded from disk instead of being constructed on the fly from "
             "cot.json reasoning fields.",
    )
    ap.add_argument("--exclude_files", nargs="*", default=[])
    ap.add_argument("--require_validation_ok", action="store_true")
    ap.add_argument("--skip_png", action="store_true",
                    help="Train ONLY on forged (.jpg) documents; skip every "
                         "clean (.png) document. In this mode the student "
                         "always sees tampered docs and learns to always "
                         "report anomalies.")
    ap.add_argument("--val_split", type=float, default=0.02)

    # Heatmap
    ap.add_argument("--use_heatmap", action="store_true",
                    help="Feed precomputed DTD heatmap as 2nd image. Requires "
                         "--heatmap_dir. Samples without a matching heatmap "
                         "are skipped. Force-disabled when "
                         "--only_second_stage_reasoning is set.")
    ap.add_argument("--heatmap_dir", default=None,
                    help="Dir with precomputed heatmap PNGs.")

    # Reasoning-mode flags (mutually exclusive — only one may be set)
    ap.add_argument("--disable_reasoning", action="store_true",
                    help="Target = augmented_gt only (no <think> block).")
    ap.add_argument("--only_first_stage_reasoning", action="store_true",
                    help="Target = STAGE 1+2 reasoning only (no report).")
    ap.add_argument("--only_second_stage_reasoning", action="store_true",
                    help="Target = STAGE 3+4 reasoning + report. Heatmap is "
                         "force-disabled to match the stage-2 prompt.")

    # Viz
    ap.add_argument("--viz_inputs", default=None,
                    help="If set, save per-sample input artifacts.")
    ap.add_argument("--viz_max_samples", type=int, default=40)

    ap.add_argument("--max_seq_length", type=int, default=0)
    ap.add_argument("--max_image_pixels", type=int, default=1048576,
                    help="Max pixels PER IMAGE. With --use_heatmap, total = 2x.")
    ap.add_argument("--optim", default="adamw_torch",
                    choices=("adamw_torch", "adamw_bnb_8bit",
                             "paged_adamw_32bit", "paged_adamw_8bit"))

    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--model_class", default="Qwen3VLForConditionalGeneration")
    ap.add_argument("--attn_impl", default="flash_attention_2",
                    choices=("eager", "sdpa", "flash_attention_2"))

    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs",       type=float, default=1.0)
    ap.add_argument("--per_device_batch_size",  type=int,   default=1)
    ap.add_argument("--grad_accum_steps",       type=int,   default=8)
    ap.add_argument("--learning_rate",          type=float, default=1e-5)
    ap.add_argument("--warmup_ratio",           type=float, default=0.03)
    ap.add_argument("--weight_decay",           type=float, default=0.0)
    ap.add_argument("--lr_scheduler_type",      default="cosine")
    ap.add_argument("--max_grad_norm",          type=float, default=1.0)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_gradient_checkpointing",
                    dest="gradient_checkpointing", action="store_false")

    ap.add_argument("--fsdp_offload", action="store_true", default=True)
    ap.add_argument("--no_fsdp_offload", dest="fsdp_offload", action="store_false")

    ap.add_argument("--logging_steps",     type=int, default=5)
    ap.add_argument("--save_steps",        type=int, default=200)
    ap.add_argument("--save_total_limit",  type=int, default=3)
    ap.add_argument("--eval_steps",        type=int, default=200)
    ap.add_argument("--seed",              type=int, default=42)
    ap.add_argument("--use_liger", action="store_true")

    ap.add_argument(
        "--resume_from_checkpoint", default=None,
        help="Path to a checkpoint dir, OR 'auto' for latest checkpoint-* "
             "under --output_dir, OR unset to start from scratch.",
    )

    return ap.parse_args()


def _resolve_resume_checkpoint(arg_value: Optional[str],
                                output_dir: Path) -> Optional[str]:
    if arg_value is None:
        return None
    rank = int(os.environ.get("RANK", "0"))

    if arg_value.lower() == "auto":
        if not output_dir.exists():
            if rank == 0:
                print(f"[resume] output_dir does not exist yet -> starting "
                      f"from scratch", flush=True)
            return None
        ckpts = []
        for p in output_dir.iterdir():
            if p.is_dir() and p.name.startswith("checkpoint-"):
                try:
                    step = int(p.name.split("-", 1)[1])
                    ckpts.append((step, p))
                except ValueError:
                    pass
        if not ckpts:
            if rank == 0:
                print(f"[resume] no checkpoint-* dirs under {output_dir}; "
                      f"starting from scratch", flush=True)
            return None
        ckpts.sort()
        latest_step, latest_path = ckpts[-1]
        if rank == 0:
            print(f"[resume] auto-resume from {latest_path} (step={latest_step})",
                  flush=True)
        return str(latest_path)

    p = Path(arg_value).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"[resume] checkpoint dir does not exist: {p}")
    if not p.is_dir():
        raise SystemExit(f"[resume] resume path must be a dir, got: {p}")
    if not (p / "trainer_state.json").exists():
        if rank == 0:
            print(f"[resume] WARN: {p}/trainer_state.json missing", flush=True)
    if rank == 0:
        print(f"[resume] resuming from explicit checkpoint: {p}", flush=True)
    return str(p)


def _prefetch_model_rank0(model_id: str) -> None:
    import torch.distributed as dist
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    rank = int(os.environ.get("RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
        _initialised_here = True
    else:
        _initialised_here = False
    try:
        if rank == 0:
            print(f"[prefetch] rank 0 downloading {model_id} ...", flush=True)
            from huggingface_hub import snapshot_download
            snapshot_download(model_id, allow_patterns=[
                "*.safetensors", "*.json", "*.txt", "tokenizer*",
                "preprocessor*", "vocab*", "merges*", "chat_template*", "*.py",
            ])
            print("[prefetch] rank 0 done", flush=True)
        else:
            print(f"[prefetch] rank {rank} waiting ...", flush=True)
        dist.barrier()
        if rank != 0:
            print(f"[prefetch] rank {rank} released", flush=True)
    finally:
        if _initialised_here:
            dist.destroy_process_group()


def main() -> int:
    from transformers import Trainer, TrainingArguments, set_seed
    from torch.utils.data import random_split

    args = parse_args()
    set_seed(args.seed)
    rank = int(os.environ.get("RANK", "0"))

    # ---- Validate mutually-exclusive reasoning-mode flags ----
    mode_flags = [
        args.disable_reasoning,
        args.only_first_stage_reasoning,
        args.only_second_stage_reasoning,
    ]
    if sum(mode_flags) > 1:
        raise SystemExit(
            "Only one of --disable_reasoning, --only_first_stage_reasoning, "
            "--only_second_stage_reasoning can be set at a time."
        )
    if args.disable_reasoning:
        reasoning_mode = "disable_reasoning"
    elif args.only_first_stage_reasoning:
        reasoning_mode = "only_first_stage_reasoning"
    elif args.only_second_stage_reasoning:
        reasoning_mode = "only_second_stage_reasoning"
    else:
        reasoning_mode = "default"

    if args.use_heatmap and not args.heatmap_dir:
        raise SystemExit(
            "--use_heatmap requires --heatmap_dir. "
            "Run scripts/precompute_heatmaps.py first to generate heatmaps."
        )

    if args.train_mode == "lora" and args.learning_rate == 1e-5:
        args.learning_rate = 1e-4
        if rank == 0:
            print(f"[lr] auto-bumped lr to {args.learning_rate} for LoRA",
                  flush=True)

    _prefetch_model_rank0(args.model_id)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_dir = Path(args.prerendered_prompts_dir).expanduser().resolve()
    if not prompts_dir.is_dir():
        raise SystemExit(
            f"--prerendered_prompts_dir not a directory: {prompts_dir}\n"
            f"Run prerender_prompts.py first to populate it."
        )
    if rank == 0:
        # Count available prompts for sanity
        n_prompts = sum(1 for _ in prompts_dir.rglob("*.prompt.txt"))
        print(f"[prompts] {prompts_dir}  ({n_prompts} pre-rendered prompts)",
              flush=True)

    targets_dir = (Path(args.prerendered_targets_dir).expanduser().resolve()
                   if args.prerendered_targets_dir else None)
    if targets_dir is not None and rank == 0:
        n_targets = sum(1 for _ in targets_dir.rglob("*.target.txt"))
        print(f"[targets] {targets_dir}  ({n_targets} pre-rendered targets)",
              flush=True)

    exclude_paths = [Path(f).expanduser().resolve() for f in args.exclude_files]
    excluded = _load_excluded_stems(exclude_paths)
    if rank == 0:
        print(f"[exclude] {len(excluded)} stems to skip from "
              f"{len(exclude_paths)} files", flush=True)

    # ---- Validate dataset BEFORE loading the model (fast fail) ----
    heatmap_dir = (Path(args.heatmap_dir).expanduser().resolve()
                   if args.heatmap_dir else None)
    if heatmap_dir is not None and rank == 0:
        n_pngs = sum(1 for _ in heatmap_dir.rglob("*.png"))
        print(f"[heatmap_dir] {heatmap_dir}  ({n_pngs} PNGs total)", flush=True)

    viz_dir = (Path(args.viz_inputs).expanduser().resolve()
               if args.viz_inputs else None)
    if viz_dir is not None and rank == 0:
        print(f"[viz] inputs will be saved to {viz_dir} "
              f"(max {args.viz_max_samples} samples)", flush=True)

    full_ds = CotJsonlDataset(
        cot_root=Path(args.cot_root).expanduser().resolve(),
        image_root=Path(args.image_root).expanduser().resolve(),
        prerendered_prompts_dir=prompts_dir,
        excluded_stems=excluded,
        require_validation_ok=args.require_validation_ok,
        use_heatmap=args.use_heatmap,
        heatmap_dir=heatmap_dir,
        prerendered_targets_dir=targets_dir,
        reasoning_mode=reasoning_mode,
        viz_dir=viz_dir,
        viz_max_samples=args.viz_max_samples,
        max_image_pixels=args.max_image_pixels,
        skip_png=args.skip_png,
    )

    # ---- Load model ONLY after dataset validation passes ----
    model, processor = build_model_and_processor(args)

    if args.val_split > 0:
        n_val = max(1, int(len(full_ds) * args.val_split))
        n_train = len(full_ds) - n_val
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, eval_ds = random_split(full_ds, [n_train, n_val], generator=gen)
        if rank == 0:
            print(f"[split] train={n_train}  val={n_val}", flush=True)
    else:
        train_ds, eval_ds = full_ds, None

    collator = MMCollator(processor=processor,
                          max_seq_length=args.max_seq_length)

    common_args = dict(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=False,  # we handle this manually
        bf16=True, tf32=True,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    if rank == 0:
        print(f"[optim] {args.optim}", flush=True)

    if args.train_mode == "full":
        layer_classes = getattr(model, "_no_split_modules", None) \
                        or ["Qwen3VLTextDecoderLayer"]
        if rank == 0:
            print(f"[fsdp] wrap layers: {layer_classes}", flush=True)
            print(f"[fsdp] optimizer CPU offload: {args.fsdp_offload}", flush=True)
        fsdp_config = {
            "fsdp_transformer_layer_cls_to_wrap": layer_classes,
            "fsdp_sync_module_states":   True,
            "fsdp_use_orig_params":      True,
            "fsdp_offload_params":       args.fsdp_offload,
            "fsdp_backward_prefetch":    "backward_pre",
            "fsdp_state_dict_type":      "SHARDED_STATE_DICT",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_activation_checkpointing": True,
        }
        targs = TrainingArguments(
            **common_args,
            fsdp="full_shard auto_wrap",
            fsdp_config=fsdp_config,
        )
    else:
        targs = TrainingArguments(**common_args)

    StepStatsCallbackCls = _make_stats_callback_class()
    stats_cb = StepStatsCallbackCls(
        output_dir=out_dir,
        print_every_n_steps=max(1, args.logging_steps // 5),
    )

    EpochCheckpointCallbackCls = _make_epoch_checkpoint_callback_class()
    epoch_cb = EpochCheckpointCallbackCls(output_dir=out_dir)

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, callbacks=[stats_cb, epoch_cb],
    )

    if rank == 0:
        eff_bs = (args.per_device_batch_size * args.grad_accum_steps
                  * int(os.environ.get("WORLD_SIZE", "1")))
        print(f"[train] mode={args.train_mode}  reasoning={reasoning_mode}  "
              f"samples={len(train_ds)}  "
              f"eval={len(eval_ds) if eval_ds else 0}  "
              f"effective_bs={eff_bs}", flush=True)

    resume_path = _resolve_resume_checkpoint(
        args.resume_from_checkpoint, out_dir,
    )

    trainer.train(resume_from_checkpoint=resume_path)

    if args.train_mode == "lora":
        adapter_dir = out_dir / "adapter_final"
        if rank == 0:
            print(f"[done] saving LoRA adapter to {adapter_dir}", flush=True)
            adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))
        if rank == 0:
            processor.save_pretrained(str(adapter_dir))
        if args.merge_after_training and rank == 0:
            merged_dir = out_dir / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            processor.save_pretrained(merged_dir)
            print(f"[merge] saved to {merged_dir}", flush=True)
    else:
        if rank == 0:
            print(f"[done] saving final model to {out_dir}/final/", flush=True)
        trainer.save_model(str(out_dir / "final"))
        if rank == 0:
            processor.save_pretrained(str(out_dir / "final"))

    return 0


if __name__ == "__main__":
    sys.exit(main())