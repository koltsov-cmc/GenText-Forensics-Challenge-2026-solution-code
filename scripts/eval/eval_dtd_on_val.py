#!/usr/bin/env python
"""Evaluate and visualize a trained DTD checkpoint on a validation split.

Computes SDet + SLoc metrics with optional TTA. Two modes: metadata mode (full
GT from RealText-V2 split) and image-dir mode (folder of images, optional GT
masks). Generates visualization panels when --save_viz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = Path(__file__).resolve().parent
_possible_roots = [_SCRIPT_DIR, _SCRIPT_DIR.parent]
_TOOLKIT_ROOT = None
for r in _possible_roots:
    if (r / "realtext_v2").is_dir():
        _TOOLKIT_ROOT = r
        break
if _TOOLKIT_ROOT is None:
    _TOOLKIT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_TOOLKIT_ROOT))

from realtext_v2 import load_metadata
from realtext_v2.metadata import resolve_paths
from realtext_v2.metrics import detection_scores

# --------------------------------------------------------------------------- #
# Re-use DTD inference machinery
# --------------------------------------------------------------------------- #
_DTD_SCRIPT_DIR = _TOOLKIT_ROOT / "ForensicHub" / "dtd_train"
sys.path.insert(0, str(_DTD_SCRIPT_DIR))

import run_doc_forensics_inference as _dtd  # noqa: E402


# --------------------------------------------------------------------------- #
# Per-image visualisation
# --------------------------------------------------------------------------- #
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _find_gt_mask(masks_dir: Path | None, stem: str) -> Path | None:
    """Look for {stem}.* or {stem}_mask.* inside masks_dir."""
    if masks_dir is None or not masks_dir.is_dir():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        for cand in (masks_dir / f"{stem}{ext}",
                     masks_dir / f"{stem}_mask{ext}"):
            if cand.exists():
                return cand
    return None


def _pixel_iou_f1(gt_mask: np.ndarray, pred_mask: np.ndarray
                  ) -> tuple[float, float]:
    """Pixel-level IoU and F1 between two binary masks (any non-zero = positive)."""
    gt = (gt_mask > 0).astype(np.uint8)
    pr = (pred_mask > 0).astype(np.uint8)
    inter = int(np.logical_and(gt, pr).sum())
    union = int(np.logical_or(gt, pr).sum())
    iou = (inter / union) if union > 0 else float(gt.sum() == 0 and pr.sum() == 0)
    tp = inter
    fp = int(np.logical_and(pr == 1, gt == 0).sum())
    fn = int(np.logical_and(pr == 0, gt == 1).sum())
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else float(gt.sum() == 0 and pr.sum() == 0)
    return float(iou), float(f1)


def _save_visualization(
    image_pil: Image.Image,
    prob: np.ndarray,
    gt_mask: Optional[np.ndarray],
    out_path: Path,
    *,
    threshold: float,
    sample_id: str,
    pred_forged: bool,
    is_forged_gt: Optional[bool],
    iou: Optional[float],
    f1: Optional[float],
) -> None:
    """4-panel composite: image | GT mask | DTD heatmap | thresholded mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_arr = np.asarray(image_pil.convert("RGB"))
    H, W = image_arr.shape[:2]

    # Figure: 4 panels in a row, each panel is the image's aspect ratio.
    # With aspect="equal" (imshow default), each axes will be ~W:H wide:tall.
    # We size the figure so the panels render at a comfortable physical size.
    panel_w_in = min(5.5, max(3.0, W / 350))  # 3.0–5.5 inches per panel
    panel_h_in = panel_w_in * H / W
    fig_w = panel_w_in * 4 + 0.6   # small inter-panel padding
    fig_h = panel_h_in + 1.2       # room for titles, suptitle, colorbar
    fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h))

    # Panel 1 — original image
    axes[0].imshow(image_arr)
    axes[0].set_title("image", fontsize=10)
    axes[0].axis("off")

    # Panel 2 — GT mask
    if gt_mask is not None:
        axes[1].imshow(image_arr, alpha=0.35)
        axes[1].imshow(gt_mask, cmap="Reds", alpha=0.65, vmin=0, vmax=255)
        title = "GT mask"
        if is_forged_gt is not None:
            title += f"  ({'FORGED' if is_forged_gt else 'AUTHENTIC'})"
    else:
        axes[1].imshow(image_arr)
        axes[1].text(0.5, 0.5, "no GT mask", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=12,
                     color="#cccccc", alpha=0.85,
                     bbox=dict(facecolor="black", alpha=0.55, pad=8))
        title = "GT mask (n/a)"
    axes[1].set_title(title, fontsize=10)
    axes[1].axis("off")

    # Panel 3 — DTD probability overlay (jet heatmap on image)
    cmap = plt.get_cmap("jet")
    heat = (cmap(prob)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * image_arr + 0.45 * heat).clip(0, 255).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"DTD prob   max={prob.max():.3f}  mean={prob.mean():.3f}",
        fontsize=10,
    )
    axes[2].axis("off")
    # Inline colorbar via inset_axes plays nicely with tight_layout.
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = axes[2].inset_axes([0.05, -0.08, 0.9, 0.04])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7)

    # Panel 4 — thresholded mask
    pred_mask = (prob >= threshold).astype(np.uint8) * 255
    axes[3].imshow(image_arr, alpha=0.35)
    axes[3].imshow(pred_mask, cmap="Greens", alpha=0.6, vmin=0, vmax=255)
    p4_title = (f"pred mask @ thr={threshold:.2f}   "
                f"verdict={'FORGED' if pred_forged else 'AUTHENTIC'}")
    if iou is not None and f1 is not None:
        p4_title += f"\nIoU={iou:.3f}  F1={f1:.3f}"
    axes[3].set_title(p4_title, fontsize=10)
    axes[3].axis("off")

    fig.suptitle(sample_id, fontsize=11, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # DTD model
    ap.add_argument("--config", required=True,
                    help="YAML training config for DTD.")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to trained DTD .pth checkpoint.")

    # Mode A: metadata
    g_meta = ap.add_argument_group("metadata mode")
    g_meta.add_argument("--root",
                        help="RealText-V2 root (contains metadata.parquet).")
    g_meta.add_argument("--split_parquet",
                        help="Parquet file listing sample_ids for the split.")

    # Mode B: image directory
    g_dir = ap.add_argument_group("image-dir mode")
    g_dir.add_argument("--image_dir",
                       help="Directory of images to evaluate / visualise.")
    g_dir.add_argument("--gt_masks_dir",
                       help="Optional GT masks dir (.png named after image stem).")
    g_dir.add_argument("--order", choices=("sequential", "random"),
                       default="sequential",
                       help="Iteration order for --image_dir.")
    g_dir.add_argument("--seed", type=int, default=42,
                       help="Random seed when --order=random.")

    # Common
    ap.add_argument("--limit", type=int, default=100,
                    help="Maximum number of images to evaluate (0 = all).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for FORGED vs AUTHENTIC "
                         "and for binary mask extraction.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--out_dir", default="eval/dtd_val")
    ap.add_argument("--save_probs", action="store_true",
                    help="Save .dtd.prob.npy for each image.")
    ap.add_argument("--save_masks", action="store_true",
                    help="Save .dtd.mask.png for each image.")
    ap.add_argument("--save_viz", action="store_true",
                    help="Save 4-panel visualisation PNG per image.")
    ap.add_argument("--no_resume", action="store_true",
                    help="Ignore existing checkpoint and start fresh.")

    # TTA
    ap.add_argument("--tta", action="store_true",
                    help="Enable test-time augmentation.")
    ap.add_argument("--tta_combine", default="min",
                    choices=("min", "mean", "median"),
                    help="How to aggregate TTA passes.")
    ap.add_argument("--tta_passes", type=int, default=4, choices=(3, 4))
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Sample-list builders for the two modes
# --------------------------------------------------------------------------- #
def _build_samples_from_metadata(args) -> list[dict]:
    """Mode A: build list of dicts with {sample_id, image_path, mask_path,
    is_forged_gt} from metadata + optional split."""
    if not args.root:
        raise SystemExit("metadata mode requires --root.")
    print("[data] loading metadata ...")
    meta = load_metadata(args.root)
    meta = resolve_paths(meta, args.root)

    if args.split_parquet:
        split_df = pd.read_parquet(args.split_parquet)
        keep_ids = set(split_df.get(
            "original_sample_id", split_df.get("sample_id", [])
        ).tolist())
        meta = meta[meta["sample_id"].isin(keep_ids)].reset_index(drop=True)
        print(f"[data] split restricted to {len(meta)} rows")

    if args.limit > 0:
        # Stratified sample: half forged, half pristine.
        n_black = (meta["type"] == "black").sum()
        n_white = (meta["type"] == "white").sum()
        forged = meta[meta["type"] == "black"].sample(
            n=min(args.limit // 2, n_black),
            random_state=args.seed,
        )
        pristine = meta[meta["type"] == "white"].sample(
            n=min(args.limit - len(forged), n_white),
            random_state=args.seed,
        )
        meta = pd.concat([forged, pristine]).reset_index(drop=True)

    samples = []
    for _, row in meta.iterrows():
        img_path = row.get("image_path")
        mask_path = row.get("mask_path")
        if img_path is None:
            continue
        gt_type = str(row.get("type", "")).lower()
        samples.append({
            "sample_id":     str(row["sample_id"]),
            "image_path":    Path(str(img_path)),
            "mask_path":     Path(str(mask_path)) if mask_path else None,
            "is_forged_gt":  gt_type.startswith("black"),
        })
    return samples


def _build_samples_from_dir(args) -> list[dict]:
    """Mode B: enumerate --image_dir, optionally pair with --gt_masks_dir.

    is_forged_gt is inferred from the GT mask if present (any non-zero = forged).
    If no GT mask, is_forged_gt is None and the sample is skipped from metrics.
    """
    img_dir = Path(args.image_dir).expanduser().resolve()
    if not img_dir.is_dir():
        raise SystemExit(f"--image_dir not found: {img_dir}")
    masks_dir = (Path(args.gt_masks_dir).expanduser().resolve()
                 if args.gt_masks_dir else None)

    paths = sorted(p for p in img_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if args.order == "random":
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(paths))
        paths = [paths[i] for i in idx]

    if args.limit > 0:
        paths = paths[: args.limit]

    samples = []
    for p in paths:
        stem = p.stem
        gt_path = _find_gt_mask(masks_dir, stem)
        is_forged_gt: Optional[bool] = None
        if gt_path is not None:
            try:
                arr = np.array(Image.open(str(gt_path)).convert("L"),
                               dtype=np.uint8)
                is_forged_gt = bool((arr > 0).any())
            except Exception:
                is_forged_gt = False
        else:
            is_forged_gt = False
        samples.append({
            "sample_id":    stem,
            "image_path":   p,
            "mask_path":    gt_path,
            "is_forged_gt": is_forged_gt,
        })
    return samples


# --------------------------------------------------------------------------- #
# Checkpoint load / save (for resumption)
# --------------------------------------------------------------------------- #
def _load_checkpoint(ckpt_path: Path) -> dict | None:
    if not ckpt_path.exists():
        return None
    try:
        return json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_checkpoint(ckpt_path: Path, **kwargs) -> None:
    ckpt_path.write_text(
        json.dumps(kwargs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    if bool(args.image_dir) == bool(args.root):
        raise SystemExit(
            "Specify exactly one of --image_dir (image-dir mode) "
            "or --root (metadata mode)."
        )

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build sample list
    # ------------------------------------------------------------------ #
    if args.image_dir:
        samples = _build_samples_from_dir(args)
        mode = "image-dir"
    else:
        samples = _build_samples_from_metadata(args)
        mode = "metadata"
    print(f"[data] mode={mode}  evaluating {len(samples)} samples")

    if not samples:
        print("[error] no samples to process")
        return 1

    # ------------------------------------------------------------------ #
    # Load DTD model once
    # ------------------------------------------------------------------ #
    print("[dtd] loading model ...")
    t0 = time.time()
    _dtd._setup_paths_and_registry()
    model, model_name, needs_dct = _dtd.build_model_and_load(
        args.config, args.checkpoint, device
    )
    print(f"[dtd] loaded {model_name} (needs_dct={needs_dct}) "
          f"in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------ #
    # Evaluation loop (with checkpointing for resumption)
    # ------------------------------------------------------------------ #
    ckpt_path = out_dir / "progress.json"

    if args.no_resume:
        ckpt = None
    else:
        ckpt = _load_checkpoint(ckpt_path)

    if ckpt:
        completed_ids = set(ckpt.get("completed_ids", []))
        y_true         = list(ckpt.get("y_true", []))
        y_pred         = list(ckpt.get("y_pred", []))
        n_forged_gt    = int(ckpt.get("n_forged_gt", 0))
        n_forged_pred  = int(ckpt.get("n_forged_pred", 0))
        n_no_gt        = int(ckpt.get("n_no_gt", 0))
        grounding_ious = list(ckpt.get("grounding_iou", []))
        grounding_f1s  = list(ckpt.get("grounding_f1", []))
        print(f"[resume] loaded checkpoint: {len(completed_ids)} already done, "
              f"{len(y_true)} with GT, {len(grounding_ious)} grounded")
    else:
        completed_ids = set()
        y_true, y_pred = [], []
        n_forged_gt, n_forged_pred, n_no_gt = 0, 0, 0
        grounding_ious: list[float] = []
        grounding_f1s:  list[float] = []

    n_total = len(samples)
    pending = [s for s in samples if s["sample_id"] not in completed_ids]
    if len(pending) < len(samples):
        print(f"[resume] {len(pending)} remaining of {n_total} total")

    for i, s in enumerate(pending):
        img_path = s["image_path"]
        mask_path = s["mask_path"]
        sample_id = s["sample_id"]
        is_forged_gt = s["is_forged_gt"]

        if not img_path.exists():
            print(f"  [{len(completed_ids)+1}/{n_total}] {sample_id}: SKIP (no image)")
            completed_ids.add(sample_id)
            _save_checkpoint(
                ckpt_path,
                n_total=n_total, completed_ids=sorted(completed_ids),
                y_true=y_true, y_pred=y_pred,
                n_forged_gt=n_forged_gt, n_forged_pred=n_forged_pred,
                n_no_gt=n_no_gt,
                grounding_iou=grounding_ious, grounding_f1=grounding_f1s,
            )
            continue

        stem = img_path.stem
        gt_str = (f"forged_gt={is_forged_gt}" if is_forged_gt is not None
                  else "forged_gt=?")
        print(f"\n[{len(completed_ids)+1}/{n_total}] {sample_id}  {gt_str}")

        # ---- DTD inference ----
        tta_offsets = _dtd._build_tta_offsets(args.tta, args.tta_passes)
        if tta_offsets is not None:
            print(f"[tta] enabled  passes={len(tta_offsets)}  "
                f"combine={args.tta_combine}  offsets={tta_offsets}")
        
        t0 = time.time()
        prob, image_pil = _dtd.infer_one_image(
            img_path, model, model_name, needs_dct, device,
            jpeg_quality=args.jpeg_quality,
            tta_offsets=tta_offsets,
            tta_combine=args.tta_combine,
        )
        print(f"  [dtd] {time.time()-t0:.1f}s  prob shape={prob.shape}  "
              f"max={float(prob.max()):.4f}  mean={float(prob.mean()):.4f}")

        # ---- Detection verdict ----
        pred_forged = float(prob.max()) > args.threshold
        if pred_forged:
            n_forged_pred += 1
        if is_forged_gt is True:
            n_forged_gt += 1

        if is_forged_gt is None:
            n_no_gt += 1
            print(f"  [det] gt=?  pred={'FORGED' if pred_forged else 'AUTHENTIC'}  (skipped from metrics)")
        else:
            gt_label = "FORGED" if is_forged_gt else "AUTHENTIC"
            pred_label = "FORGED" if pred_forged else "AUTHENTIC"
            y_true.append(gt_label)
            y_pred.append(pred_label)
            print(f"  [det] gt={gt_label}  pred={pred_label}")

        # ---- Save raw prob / binary mask ----
        if args.save_probs:
            np.save(out_dir / f"{stem}.dtd.prob.npy", prob.astype(np.float32))
        pred_mask = ((prob >= args.threshold) * 255).astype(np.uint8)
        if args.save_masks:
            Image.fromarray(pred_mask, mode="L").save(
                out_dir / f"{stem}.dtd.mask.png"
            )

        # ---- GT mask (resized to image) ----
        img_w, img_h = image_pil.size
        gt_mask: Optional[np.ndarray] = None
        if mask_path is not None and Path(str(mask_path)).exists():
            m = Image.open(str(mask_path)).convert("L")
            if m.size != (img_w, img_h):
                m = m.resize((img_w, img_h), Image.NEAREST)
            gt_mask = np.array(m, dtype=np.uint8)
            gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        # ---- Per-image IoU/F1 (only if we have GT) ----
        iou_i: Optional[float] = None
        f1_i: Optional[float] = None
        if gt_mask is not None and is_forged_gt is not None:
            iou_i, f1_i = _pixel_iou_f1(gt_mask, pred_mask)
            grounding_ious.append(iou_i)
            grounding_f1s.append(f1_i)
            print(f"  [loc] iou={iou_i:.4f}  f1={f1_i:.4f}")

        # ---- Visualisation ----
        if args.save_viz:
            _save_visualization(
                image_pil=image_pil,
                prob=prob,
                gt_mask=gt_mask,
                out_path=viz_dir / f"{stem}.viz.png",
                threshold=args.threshold,
                sample_id=sample_id,
                pred_forged=pred_forged,
                is_forged_gt=is_forged_gt,
                iou=iou_i,
                f1=f1_i,
            )

        # ---- Checkpoint ----
        completed_ids.add(sample_id)
        _save_checkpoint(
            ckpt_path,
            n_total=n_total, completed_ids=sorted(completed_ids),
            y_true=y_true, y_pred=y_pred,
            n_forged_gt=n_forged_gt, n_forged_pred=n_forged_pred,
            n_no_gt=n_no_gt,
            grounding_iou=grounding_ious, grounding_f1=grounding_f1s,
        )

    # ------------------------------------------------------------------ #
    # Compute aggregate metrics (only over samples with GT)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    out_dict: dict = {
        "mode": mode,
        "n_total": int(n_total),
        "n_with_gt": len(y_true),
        "n_no_gt": n_no_gt,
        "threshold": args.threshold,
        "forged_gt_count": n_forged_gt,
        "forged_pred_count": n_forged_pred,
    }

    if y_true:
        det = detection_scores(y_true, y_pred)
        print(f"\n[Detection (SDet)]")
        print(f"  precision = {det['precision']:.4f}")
        print(f"  recall    = {det['recall']:.4f}")
        print(f"  f1        = {det['f1']:.4f}")
        print(f"  accuracy  = {det['accuracy']:.4f}")
        print(f"  n         = {det['n']}")
        print(f"  forged_gt = {n_forged_gt}  forged_pred = {n_forged_pred}")
        out_dict["detection"] = det
    else:
        print("\n[Detection] skipped — no GT labels available.")

    if grounding_ious:
        ious_arr = np.array(grounding_ious, dtype=np.float64)
        f1s_arr  = np.array(grounding_f1s,  dtype=np.float64)

        mIoU = float(ious_arr.mean())
        mF1  = float(f1s_arr.mean())

        loc = {
            "mIoU": mIoU,
            "mF1":  mF1,
            "mIoU_forged_only": mIoU,
            "mF1_forged_only":  mF1,
            "n":    len(grounding_ious),
            "n_forged": n_forged_gt,
        }
        print(f"\n[Grounding (SLoc)]")
        print(f"  mIoU            = {mIoU:.4f}")
        print(f"  mF1             = {mF1:.4f}")
        print(f"  n               = {loc['n']}  (forged={n_forged_gt})")

        s_loc = 0.5 * mIoU + 0.5 * mF1
        print(f"\n  SLoc composite (0.5*mIoU + 0.5*mF1) = {s_loc:.4f}")
        out_dict["grounding"] = loc
        out_dict["s_loc_composite"] = float(s_loc)
    else:
        print("\n[Grounding] skipped — no GT masks available.")

    # Remove progress checkpoint on successful completion
    if ckpt_path.exists():
        ckpt_path.unlink()
        print("\n[resume] progress checkpoint removed (completed).")

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    out_path = out_dir / "eval.json"
    out_path.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
    print(f"\n[done] results saved to {out_path}")
    if args.save_viz:
        print(f"       visualisations in {viz_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())