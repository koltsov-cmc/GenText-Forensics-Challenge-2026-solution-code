"""Random-access dataset interface over the downloaded RealText-V2 files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image

from .metadata import load_metadata, resolve_paths
from .report import ForgeryReport, parse_report


@dataclass
class Sample:
    sample_id: str
    language: str
    language_code: str
    type: str                          # "black" (forged) / "white" (pristine)
    image_path: Optional[Path]
    mask_path: Optional[Path]
    report_path: Optional[Path]
    report_text: str = ""

    # ------- lazy loaders -------
    def image(self) -> Image.Image:
        if self.image_path is None:
            raise FileNotFoundError(f"No image on disk for {self.sample_id}")
        return Image.open(self.image_path).convert("RGB")

    def mask(self) -> Optional[np.ndarray]:
        """Return mask as uint8 [H, W] (0 / 255) or ``None`` for pristine."""
        if self.mask_path is None:
            return None
        m = Image.open(self.mask_path).convert("L")
        return np.array(m, dtype=np.uint8)

    def report(self) -> ForgeryReport:
        text = self.report_text
        if not text and self.report_path is not None:
            text = Path(self.report_path).read_text(encoding="utf-8")
        return parse_report(text)

    @property
    def is_forged(self) -> bool:
        return str(self.type).lower() == "black"


class RealTextV2Dataset:
    """Random-access wrapper over the dataset.

    Usage::

        ds = RealTextV2Dataset("/data/RealText-V2")
        print(len(ds))
        sample = ds[0]
        img = sample.image()
        mask = sample.mask()
        rep = sample.report()

        # Filter
        forged = ds.filter(type="black", language_code="en")
    """

    def __init__(
        self,
        root: str | Path,
        *,
        metadata: Optional[pd.DataFrame] = None,
        resolve: bool = True,
        strict: bool = False,
    ):
        self.root = Path(root)
        if metadata is None:
            metadata = load_metadata(self.root)
        if resolve and "image_path" not in metadata.columns:
            metadata = resolve_paths(metadata, self.root)

        if strict:
            # Drop rows whose image is missing on disk.
            mask = metadata["image_path"].notna()
            metadata = metadata[mask].reset_index(drop=True)

        self.meta = metadata.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.meta)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key: Union[int, str, slice, Sequence[int]]):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        if isinstance(key, (list, tuple, np.ndarray, pd.Index)):
            return [self[int(i)] for i in key]
        if isinstance(key, str):
            matches = self.meta.index[self.meta["sample_id"] == key]
            if len(matches) == 0:
                raise KeyError(key)
            key = int(matches[0])
        return self._row_to_sample(self.meta.iloc[int(key)])

    # ------------------------------------------------------------------ #
    def filter(self, **conds) -> "RealTextV2Dataset":
        """Return a new dataset view keeping rows matching all conditions.

        Each condition may be a scalar or an iterable of accepted values.
        """
        m = pd.Series(True, index=self.meta.index)
        for col, val in conds.items():
            if col not in self.meta.columns:
                raise KeyError(f"Unknown column: {col}")
            if isinstance(val, (list, tuple, set)):
                m &= self.meta[col].isin(list(val))
            else:
                m &= self.meta[col] == val
        return RealTextV2Dataset(
            self.root,
            metadata=self.meta[m].reset_index(drop=True),
            resolve=False,
        )

    def sample(self, n: int = 1, random_state: Optional[int] = None) -> "RealTextV2Dataset":
        sub = self.meta.sample(
            n=min(n, len(self)),
            random_state=random_state,
        ).reset_index(drop=True)
        return RealTextV2Dataset(self.root, metadata=sub, resolve=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_sample(row: pd.Series) -> Sample:
        def _p(key):
            v = row.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            return Path(v) if not isinstance(v, Path) else v

        return Sample(
            sample_id=str(row.get("sample_id", "")),
            language=str(row.get("language", "")),
            language_code=str(row.get("language_code", "")),
            type=str(row.get("type", "")),
            image_path=_p("image_path"),
            mask_path=_p("mask_path"),
            report_path=_p("report_path"),
            report_text=str(row.get("report_text", "") or ""),
        )
