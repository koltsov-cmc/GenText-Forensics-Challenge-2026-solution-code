"""Subprocess wrapper for PaddleOCR.

Runs run_paddle_sobel.py as a separate process with CUDA_VISIBLE_DEVICES set
to the desired GPU. This guarantees PaddleOCR cannot interfere with the
main process's CUDA context (which is owned by Qwen via device_map="auto").

Usage:
    from run_paddle_subprocess import run_paddle_ocr_subprocess
    
    result = run_paddle_ocr_subprocess(
        image_path="/path/to/img.jpg",
        candidate_langs=["en", "ch"],
        gpu_id=7,
    )

The result has the same schema as run_paddle_ocr_with_lang_detect().
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent


def run_paddle_ocr_subprocess(
    image_path: str | Path,
    *,
    candidate_langs: list[str],
    gpu_id: int = 7,
    mag_ratio: float = 1.0,
    timeout_sec: float = 300.0,
    script_path: str | Path | None = None,
) -> dict:
    """Run run_paddle_sobel.py in a subprocess, isolated on a single GPU.

    Returns the OCR result dict (same schema as run_paddle_ocr_with_lang_detect).
    """
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    # Locate the script
    if script_path is None:
        script_path = _THIS_DIR / "run_paddle_sobel.py"
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"run_paddle_sobel.py not found at {script_path}")

    # Isolate paddle to one GPU via env var. This is the only reliable way
    # to prevent paddle from grabbing the same GPU as torch's device_map.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Limit paddle's GPU memory pool to ~5% — prevents it from grabbing all 80GB
    # on the visible GPU, which matters if other processes share the same card.
    env["FLAGS_fraction_of_gpu_memory_to_use"] = "0.05"
    env["FLAGS_allocator_strategy"] = "auto_growth"

    # Output to temp file (subprocess can't return Python objects)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        out_json = f.name

    try:
        cmd = [
            sys.executable, str(script_path),
            "--image", str(image_path),
            "--langs", ",".join(candidate_langs),
            "--mag_ratio", str(mag_ratio),
            "--out_json", out_json,
            "--quiet",
        ]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"OCR subprocess failed (rc={result.returncode}):\n"
                f"STDOUT: {result.stdout[-1000:]}\n"
                f"STDERR: {result.stderr[-1000:]}"
            )

        with open(out_json, "r", encoding="utf-8") as f:
            ocr_result = json.load(f)
        return ocr_result

    finally:
        # Cleanup temp file
        try:
            os.unlink(out_json)
        except OSError:
            pass