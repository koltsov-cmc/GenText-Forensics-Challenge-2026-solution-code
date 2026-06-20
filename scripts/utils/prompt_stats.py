#!/usr/bin/env python
"""Compute token-length statistics over a directory of pre-rendered prompts.

Reports char/token counts, percentiles, and histogram for .prompt.txt files
in {prompts_dir}/partXXX/ layout. Supports parallel tokenisation.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path


# Global handle so worker processes don't re-create the tokeniser per file
_TOKENIZER = None
_TOKENIZER_ID = None


def _init_worker(tokenizer_id: str) -> None:
    global _TOKENIZER, _TOKENIZER_ID
    if _TOKENIZER is None or _TOKENIZER_ID != tokenizer_id:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True,
        )
        _TOKENIZER_ID = tokenizer_id


def _tokenise(path_str: str) -> tuple[str, int, int]:
    """Return (path, n_chars, n_tokens) for one file."""
    text = Path(path_str).read_text(encoding="utf-8", errors="replace")
    n_chars = len(text)
    # `add_special_tokens=False` → just raw token count of the prompt body,
    # without the chat-template scaffold (which gets added later in training).
    ids = _TOKENIZER(text, add_special_tokens=False)["input_ids"]
    return path_str, n_chars, len(ids)


def _gather_files(prompts_dir: Path) -> list[Path]:
    out: list[Path] = []
    for part_dir in sorted(prompts_dir.iterdir()):
        if part_dir.is_dir() and part_dir.name.lower().startswith("part"):
            out.extend(sorted(part_dir.glob("*.prompt.txt")))
    # Fallback: flat layout
    if not out:
        out = sorted(prompts_dir.glob("*.prompt.txt"))
    return out


def _percentiles(sorted_values: list[int], ps: list[float]) -> list[int]:
    """Linear percentiles of a pre-sorted list."""
    if not sorted_values:
        return [0] * len(ps)
    n = len(sorted_values)
    out = []
    for p in ps:
        if p >= 100.0:
            out.append(sorted_values[-1])
            continue
        idx = max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))
        out.append(sorted_values[idx])
    return out


def _histogram(values: list[int], n_bins: int = 12, width: int = 50) -> str:
    """Build a simple text histogram of token counts."""
    if not values:
        return "(empty)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"  [{lo:>7d}]: {'#' * width}  ({len(values)})"
    bin_w = max(1, (hi - lo + n_bins - 1) // n_bins)
    bins = [0] * n_bins
    for v in values:
        b = min(n_bins - 1, (v - lo) // bin_w)
        bins[b] += 1
    max_count = max(bins) or 1
    lines = []
    for i, count in enumerate(bins):
        rng_lo = lo + i * bin_w
        rng_hi = rng_lo + bin_w - 1
        bar_len = int(count * width / max_count)
        lines.append(
            f"  [{rng_lo:>7d}..{rng_hi:<7d}]: "
            f"{'#' * bar_len:<{width}}  {count}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts_dir", required=True,
                    help="Directory with partXXX/{stem}.prompt.txt files.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-VL-32B-Instruct",
                    help="HuggingFace tokeniser id (default: Qwen3-VL-32B).")
    ap.add_argument("--workers", type=int, default=0,
                    help="Worker processes. 0 = max(1, os.cpu_count()//2).")
    ap.add_argument("--over", type=int, default=0,
                    help="If >0, list files whose token count exceeds this.")
    ap.add_argument("--top", type=int, default=15,
                    help="How many top --over files to print (default 15).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N files (for sanity).")
    args = ap.parse_args()

    prompts_dir = Path(args.prompts_dir).expanduser().resolve()
    if not prompts_dir.is_dir():
        raise SystemExit(f"prompts_dir not a directory: {prompts_dir}")

    files = _gather_files(prompts_dir)
    if not files:
        raise SystemExit(f"no *.prompt.txt files under {prompts_dir}")
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[scan] {len(files)} *.prompt.txt files under {prompts_dir}",
          flush=True)
    print(f"[tokenizer] {args.tokenizer}", flush=True)

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) // 2)
    print(f"[workers] {workers}", flush=True)

    t0 = time.time()
    results: list[tuple[str, int, int]] = []
    if workers <= 1:
        _init_worker(args.tokenizer)
        for i, p in enumerate(files, 1):
            results.append(_tokenise(str(p)))
            if i % 200 == 0:
                print(f"  [{i}/{len(files)}] elapsed={time.time()-t0:.1f}s",
                      flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(args.tokenizer,),
        ) as pool:
            for i, r in enumerate(
                pool.imap_unordered(_tokenise, [str(p) for p in files],
                                     chunksize=16), 1
            ):
                results.append(r)
                if i % 200 == 0:
                    print(f"  [{i}/{len(files)}] elapsed={time.time()-t0:.1f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"[done] tokenised {len(results)} files in {elapsed:.1f}s "
          f"({len(results)/max(elapsed,1e-9):.0f} files/s)", flush=True)
    print()

    chars   = [r[1] for r in results]
    tokens  = [r[2] for r in results]
    tokens_sorted = sorted(tokens)
    chars_sorted  = sorted(chars)

    ps_list = [50, 75, 90, 95, 99, 99.5, 100]
    char_pcts  = _percentiles(chars_sorted,  ps_list)
    token_pcts = _percentiles(tokens_sorted, ps_list)

    print("=" * 64)
    print(f"  CHARS    total={sum(chars):>12,}  mean={statistics.mean(chars):>8.0f}  "
          f"median={statistics.median(chars):>8.0f}  min={min(chars):>6,}  max={max(chars):>7,}")
    print(f"  TOKENS   total={sum(tokens):>12,}  mean={statistics.mean(tokens):>8.0f}  "
          f"median={statistics.median(tokens):>8.0f}  min={min(tokens):>6,}  max={max(tokens):>7,}")
    print()
    print("  Percentiles    " + "  ".join(f"p{p:<5}" for p in ps_list))
    print("    chars:       " + "  ".join(f"{v:<6,}" for v in char_pcts))
    print("    tokens:      " + "  ".join(f"{v:<6,}" for v in token_pcts))
    print()
    print("Token-count distribution:")
    print(_histogram(tokens))
    print()

    if args.over > 0:
        over = [(p, c, t) for (p, c, t) in results if t > args.over]
        over.sort(key=lambda r: -r[2])
        print(f"Files with tokens > {args.over}: {len(over)} "
              f"({100.0 * len(over) / len(results):.2f}%)")
        for p, c, t in over[: args.top]:
            print(f"  tokens={t:<7,}  chars={c:<8,}  {p}")
        if len(over) > args.top:
            print(f"  ... and {len(over) - args.top} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())