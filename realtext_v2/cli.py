"""Command-line helpers. Run ``python -m realtext_v2 --help``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .download import download_dataset, download_metadata_only
from .metadata import load_metadata, metadata_stats
from .dataset import RealTextV2Dataset
from .splits import stratified_split, split_report
from .viz import save_sample_figure
from .vlm_format import export_sft_jsonl


def _cmd_download(args):
    if args.metadata_only:
        path = download_metadata_only(args.root, token=args.token)
    else:
        parts = [int(p) for p in args.parts.split(",")] if args.parts else None
        langs = args.languages.split(",") if args.languages else None
        path = download_dataset(
            args.root,
            parts=parts,
            languages=langs,
            include_images=not args.no_images,
            include_masks=not args.no_masks,
            include_reports=not args.no_reports,
            token=args.token,
        )
    print(f"Downloaded to: {path}")


def _cmd_stats(args):
    meta = load_metadata(args.root)
    stats = metadata_stats(meta)
    print(json.dumps(stats, indent=2, default=str))


def _cmd_show(args):
    ds = RealTextV2Dataset(args.root, strict=True)
    if args.sample_id:
        s = ds[args.sample_id]
    else:
        s = ds[args.index]
    out = Path(args.out)
    save_sample_figure(s, out)
    print(f"Saved preview to {out}")


def _cmd_split(args):
    meta = load_metadata(args.root)
    train, val = stratified_split(
        meta,
        val_size=args.val_size,
        random_state=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet")
    val.to_parquet(out / "val.parquet")
    with (out / "split_report.json").open("w", encoding="utf-8") as f:
        json.dump(split_report(train, val), f, indent=2, default=str)
    print(f"Wrote {out/'train.parquet'}  ({len(train)} rows)")
    print(f"Wrote {out/'val.parquet'}    ({len(val)} rows)")


def _cmd_export(args):
    meta = load_metadata(args.root)
    if args.split:
        sp = Path(args.split)
        meta = __import__("pandas").read_parquet(sp)
    ds = RealTextV2Dataset(args.root, metadata=meta, strict=True)
    n = export_sft_jsonl(
        ds,
        args.out,
        target_style=args.style,
        image_placeholder=args.image_placeholder or None,
    )
    print(f"Wrote {n} records to {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="realtext_v2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="Download dataset (or subset)")
    p.add_argument("root")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--parts", help="Comma-separated part indices, e.g. 0,1,2")
    p.add_argument("--languages", help="Comma-separated ISO codes, e.g. en,zh")
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--no-masks", action="store_true")
    p.add_argument("--no-reports", action="store_true")
    p.add_argument("--token", default=None)
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("stats", help="Print dataset stats")
    p.add_argument("root")
    p.set_defaults(func=_cmd_stats)

    p = sub.add_parser("show", help="Render a sample to PNG")
    p.add_argument("root")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--sample-id")
    p.add_argument("--out", default="sample.png")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("split", help="Make a stratified train/val split")
    p.add_argument("root")
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="splits")
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("export-sft", help="Export JSONL for VLM SFT")
    p.add_argument("root")
    p.add_argument("--out", required=True)
    p.add_argument("--split", help="Optional parquet with split metadata")
    p.add_argument("--style", choices=["dataset", "submission"], default="dataset")
    p.add_argument("--image-placeholder", default="", help='e.g. "<image>" for LLaVA')
    p.set_defaults(func=_cmd_export)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
