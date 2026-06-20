"""Download helpers for the RealText-V2 dataset from the Hugging Face Hub.

The full dataset is ~13.7 GB. You can download:

* Metadata only  (~65 MB)  -- cheap, useful to inspect structure first.
* A single part  (~1 GB each) -- good for prototyping.
* By language    -- filter samples via metadata, then download only the
                    parts that contain them.
* Everything.

Downloads are resumable and use the local HF cache.  Set the env var
``HF_HOME`` to change the cache location.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "vankey/RealText-V2"
REPO_TYPE = "dataset"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def download_metadata_only(
    local_dir: str | os.PathLike,
    repo_id: str = DEFAULT_REPO_ID,
    token: Optional[str] = None,
) -> Path:
    """Download only the metadata files (parquet + csv + README + doc_sample).

    Returns the local path to the snapshot directory.
    """
    local_dir = Path(local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        allow_patterns=[
            "metadata.parquet",
            "metadata.csv",
            "README.md",
            "doc_sample.png",
            ".gitattributes",
        ],
        token=token,
    )
    return Path(path)


def download_dataset(
    local_dir: str | os.PathLike,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    parts: Optional[Sequence[int]] = None,
    include_images: bool = True,
    include_masks: bool = True,
    include_reports: bool = True,
    languages: Optional[Sequence[str]] = None,
    token: Optional[str] = None,
    max_workers = 2
) -> Path:
    """Download a (possibly partial) snapshot of RealText-V2.

    Parameters
    ----------
    local_dir:
        Destination directory. Will be created if missing.
    repo_id:
        HF repo id. Defaults to ``vankey/RealText-V2``.
    parts:
        Iterable of part indices (e.g. ``[0, 1, 2]``) to download.
        If ``None``, **all** parts are fetched (this is the full 13.7 GB).
    include_images / include_masks / include_reports:
        Toggle subfolders.
    languages:
        Iterable of ISO 639-1 codes. If provided, the metadata is
        consulted to figure out which parts contain samples of these
        languages, and only those parts are downloaded. Requires metadata
        to be already present locally *or* downloads it first.
    token:
        Optional HF token (only needed for private repos).

    Returns
    -------
    Path to the snapshot directory.
    """
    local_dir = Path(local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    # Resolve part set from language filter if requested.
    if languages is not None:
        # Ensure metadata is available.
        meta_path = local_dir / "metadata.parquet"
        if not meta_path.exists():
            download_metadata_only(local_dir, repo_id=repo_id, token=token)
        from .metadata import load_metadata  # local import to avoid cycle
        meta = load_metadata(local_dir)
        wanted = meta[meta["language_code"].isin(list(languages))]
        lang_parts = _infer_parts_from_metadata(wanted)
        if parts is None:
            parts = sorted(lang_parts)
        else:
            parts = sorted(set(parts) & lang_parts)

    patterns: list[str] = [
        "metadata.parquet",
        "metadata.csv",
        "README.md",
        "doc_sample.png",
    ]

    subdirs = []
    if include_images:
        subdirs.append("image")
    if include_masks:
        subdirs.append("mask")
    if include_reports:
        subdirs.append("report")

    if parts is None:
        for sub in subdirs:
            patterns.append(f"train/{sub}/**")
    else:
        for sub in subdirs:
            for p in parts:
                patterns.append(f"train/{sub}/part{p:03d}/**")

    path = snapshot_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        allow_patterns=patterns,
        token=token,
        max_workers=max_workers,
    )
    return Path(path)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _infer_parts_from_metadata(meta) -> set[int]:
    """Given a metadata DataFrame, return the set of part indices that
    contain its samples. Works by inspecting the ``image_file`` / report
    paths if they include the part prefix, else by index (files are
    sharded at 1000 per part)."""
    import re
    parts: set[int] = set()
    # Strategy 1: look inside path-like fields.
    for col in ("image_file", "report_file", "mask_file"):
        if col not in meta.columns:
            continue
        for val in meta[col].dropna().astype(str):
            m = re.search(r"part(\d{3})", val)
            if m:
                parts.add(int(m.group(1)))
    if parts:
        return parts

    # Strategy 2: derive from sample_id index (1000 per part).
    if "sample_id" in meta.columns:
        for sid in meta["sample_id"].astype(str):
            m = re.search(r"(\d+)$", sid)
            if m:
                parts.add(int(m.group(1)) // 1000)
    return parts
