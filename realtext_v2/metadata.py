"""Load and summarise the RealText-V2 metadata table."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


EXPECTED_COLUMNS = {
    "sample_id",
    "language",
    "language_code",
    "type",
    "image_file",
    "mask_file",
    "has_mask",
    "report_file",
    "report_text",
}


def load_metadata(
    root: str | Path,
    *,
    prefer: str = "parquet",
) -> pd.DataFrame:
    """Load metadata.parquet (preferred) or metadata.csv.

    Parameters
    ----------
    root:
        Dataset root (the snapshot directory).
    prefer:
        ``"parquet"`` (default, faster) or ``"csv"``.
    """
    root = Path(root)
    pq = root / "metadata.parquet"
    cs = root / "metadata.csv"

    if prefer == "parquet" and pq.exists():
        df = pd.read_parquet(pq)
    elif cs.exists():
        df = pd.read_csv(cs)
    elif pq.exists():
        df = pd.read_parquet(pq)
    else:
        raise FileNotFoundError(
            f"Neither {pq} nor {cs} found. "
            "Run download_metadata_only(root) first."
        )

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        # Not fatal -- schema may evolve -- just warn via attribute.
        df.attrs["missing_columns"] = sorted(missing)

    # Normalise types.
    if "has_mask" in df.columns:
        df["has_mask"] = df["has_mask"].astype(bool)
    if "type" in df.columns:
        df["type"] = df["type"].astype("category")
    if "language_code" in df.columns:
        df["language_code"] = df["language_code"].astype("category")

    return df


def metadata_stats(meta: pd.DataFrame) -> dict:
    """Return a dict summarising class / language distribution."""
    out: dict = {"total": len(meta)}
    if "type" in meta.columns:
        out["by_type"] = meta["type"].value_counts(dropna=False).to_dict()
    if "language_code" in meta.columns:
        out["by_language"] = (
            meta["language_code"].value_counts(dropna=False).to_dict()
        )
        if "type" in meta.columns:
            out["by_language_type"] = (
                meta.groupby(["language_code", "type"], observed=True)
                .size()
                .unstack(fill_value=0)
                .to_dict(orient="index")
            )
    if "has_mask" in meta.columns:
        out["with_mask"] = int(meta["has_mask"].sum())
    return out


def resolve_paths(
    meta: pd.DataFrame,
    root: str | Path,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Attach absolute local paths for image / mask / report columns.

    Adds ``image_path``, ``mask_path``, ``report_path`` columns.  Paths
    are resolved by searching ``train/<kind>/part*/<file>``. Missing
    files yield ``None``.
    """
    root = Path(root)
    if not inplace:
        meta = meta.copy()

    # Pre-build an index: filename -> path for each kind.
    indices: dict[str, dict[str, Path]] = {}
    for kind in ("image", "mask", "report"):
        base = root / "train" / kind
        idx: dict[str, Path] = {}
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file():
                    idx[p.name] = p
        indices[kind] = idx

    def _lookup(kind: str, fname):
        if not fname or (isinstance(fname, float) and pd.isna(fname)):
            return None
        return indices[kind].get(str(fname))

    meta["image_path"] = meta.get("image_file", pd.Series([None] * len(meta))).map(
        lambda x: _lookup("image", x)
    )
    meta["mask_path"] = meta.get("mask_file", pd.Series([None] * len(meta))).map(
        lambda x: _lookup("mask", x)
    )
    meta["report_path"] = meta.get("report_file", pd.Series([None] * len(meta))).map(
        lambda x: _lookup("report", x)
    )
    return meta
