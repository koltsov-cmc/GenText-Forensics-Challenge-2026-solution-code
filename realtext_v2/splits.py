"""Stratified train / val split for RealText-V2.

By default we stratify on the (language_code, type) joint distribution
so the split preserves:

* class balance  (forged vs pristine)
* language balance  (en / zh / th / ms / id / ar)

This matters because languages are very imbalanced (e.g. Arabic has only
500 forged samples).  A naive random split would drift.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    meta: pd.DataFrame,
    *,
    val_size: float = 0.1,
    stratify_on: Iterable[str] = ("language_code", "type"),
    random_state: int = 42,
    min_val_per_group: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split metadata into (train, val) DataFrames.

    Groups too small for ``train_test_split`` (fewer than
    ``min_val_per_group`` samples) are kept entirely in train.

    Returns two DataFrames with reset indexes.
    """
    stratify_on = list(stratify_on)
    for col in stratify_on:
        if col not in meta.columns:
            raise KeyError(f"Stratify column missing: {col}")

    # Build a joint key.
    key = meta[stratify_on].astype(str).agg("|".join, axis=1)

    # Find groups too small to split.
    counts = key.value_counts()
    splittable_keys = counts[counts >= max(2, min_val_per_group * 2)].index
    splittable_mask = key.isin(splittable_keys)

    tr_idx: list[int] = []
    va_idx: list[int] = []

    # Non-splittable groups go entirely to train.
    tr_idx.extend(meta.index[~splittable_mask].tolist())

    if splittable_mask.any():
        sub = meta[splittable_mask]
        sub_key = key[splittable_mask]
        tr, va = train_test_split(
            sub.index,
            test_size=val_size,
            stratify=sub_key,
            random_state=random_state,
        )
        tr_idx.extend(tr.tolist())
        va_idx.extend(va.tolist())

    train = meta.loc[sorted(tr_idx)].reset_index(drop=True)
    val = meta.loc[sorted(va_idx)].reset_index(drop=True)
    return train, val


def split_report(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Return a dict summarising the split for logging."""
    def _group(df):
        if {"language_code", "type"}.issubset(df.columns):
            return (
                df.groupby(["language_code", "type"], observed=True)
                .size()
                .unstack(fill_value=0)
                .to_dict(orient="index")
            )
        return {}

    return {
        "train_total": len(train),
        "val_total": len(val),
        "train_groups": _group(train),
        "val_groups": _group(val),
    }
