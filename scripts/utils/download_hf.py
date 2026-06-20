#!/usr/bin/env python
"""Download RealText-V2 test split from HuggingFace Hub via snapshot_download."""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output_dir", required=True,
                    help="Target directory where files will be saved.")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.", file=sys.stderr)
        print("Install it with:  pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] repo=vankey/RealText-V2  subset=test/  ->  {out_dir}")
    snapshot_download(
        repo_id="vankey/RealText-V2",
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=["test/**"],
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[done] test files saved to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
