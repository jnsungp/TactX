#!/usr/bin/env python3
"""Truncate paired tactile sequences to the rising contact phase.

Each episode has a 0->1->0 contact pattern. For each pair we cut at
``min(peak_end_a, peak_end_b)`` where ``peak_end`` is the last frame that
is still within ``eps`` of the per-sensor peak (i.e. end of the plateau).
The truncated label is written to ``--label_out`` and the corresponding
latent frame files (for both sensors of each pair) are copied/linked
under ``--latent_out``.

python scripts/build_concat_dataset.py  
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PAIRS = {
    "de_pair": ("daimon", "eflesh"),
    "ef_pair": ("eflesh", "flexi"),
    "fd_pair": ("flexi", "daimon"),
}

PER_FRAME_KEYS = (
    "frames",
    "scores_a",
    "scores_b",
    "norm_a",
    "norm_b",
    "contact_a",
    "contact_b",
    "contact_pair",
)


def peak_end(norm: np.ndarray, eps: float) -> int:
    """Last index that is still within ``eps`` of the peak value.

    For a 0->1->0 signal this is the final frame of the peak plateau —
    i.e. the frame right before the sustained descent begins.
    """
    norm = np.asarray(norm, dtype=np.float32)
    peak = float(norm.max())
    if peak <= 0:
        return len(norm) - 1
    threshold = peak * (1.0 - eps)
    above = np.where(norm >= threshold)[0]
    return int(above[-1])


def truncate_label(label: dict, cut: int) -> dict:
    n = cut + 1
    out = dict(label)
    T = len(label["frames"])
    for k, v in label.items():
        if k in PER_FRAME_KEYS and hasattr(v, "__len__") and len(v) == T:
            out[k] = v[:n]
    return out


def place_file(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "symlink":
        dst.symlink_to(src.resolve())
    elif link_mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def process_episode(
    pair: str,
    obj: str,
    ep_file: Path,
    label_out_root: Path,
    latent_in_root: Path,
    latent_out_root: Path,
    eps: float,
    link_mode: str,
) -> tuple[int, int, bool]:
    label = torch.load(ep_file, map_location="cpu", weights_only=False)
    norm_a = label["norm_a"]
    norm_b = label["norm_b"]
    n_original = len(label["frames"])

    has_contact = bool(label["contact_pair"].any())
    if not has_contact:
        # No rising phase to keep — skip but report.
        return n_original, 0, False

    cut = min(peak_end(norm_a, eps), peak_end(norm_b, eps))

    label_out_dir = label_out_root / pair / obj
    label_out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(truncate_label(label, cut), label_out_dir / ep_file.name)

    ep_stem = ep_file.stem  # e.g. "ep0000"
    sensor_a, sensor_b = PAIRS[pair]
    for sensor in (sensor_a, sensor_b):
        src_dir = latent_in_root / pair / sensor / obj
        dst_dir = latent_out_root / pair / sensor / obj
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fr in range(cut + 1):
            fname = f"{ep_stem}_fr{fr:04d}.pt"
            src = src_dir / fname
            if not src.exists():
                raise FileNotFoundError(f"Missing latent frame: {src}")
            place_file(src, dst_dir / fname, link_mode)

    return n_original, cut + 1, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_in", default="contact_labels/dataset_final")
    ap.add_argument("--label_out", default="contact_labels/dataset_final_concat")
    ap.add_argument(
        "--latent_in",
        default="tactile_benchmark/data/latent_data_new_processed",
    )
    ap.add_argument(
        "--latent_out",
        default="tactile_benchmark/data/latent_data_new_processed_concat",
    )
    ap.add_argument("--eps", type=float, default=0.05,
                    help="plateau tolerance: cut after last frame >= peak*(1-eps)")
    ap.add_argument("--link_mode", choices=["symlink", "hardlink", "copy"],
                    default="symlink")
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS.keys()))
    ap.add_argument("--objects", nargs="*", default=None,
                    help="restrict to these object names (default: all)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    label_in_root = Path(args.label_in)
    label_out_root = Path(args.label_out)
    latent_in_root = Path(args.latent_in)
    latent_out_root = Path(args.latent_out)

    total_eps = 0
    total_kept = 0
    total_orig_frames = 0
    total_new_frames = 0
    skipped: list[str] = []

    for pair in args.pairs:
        pair_root = label_in_root / pair
        if not pair_root.exists():
            print(f"[skip] {pair}: {pair_root} not found")
            continue
        objects = sorted(p.name for p in pair_root.iterdir() if p.is_dir())
        if args.objects:
            objects = [o for o in objects if o in args.objects]
        for obj in objects:
            ep_files = sorted((pair_root / obj).glob("ep*.pt"))
            for ep in tqdm(ep_files, desc=f"{pair}/{obj}", leave=False):
                total_eps += 1
                if args.dry_run:
                    lbl = torch.load(ep, map_location="cpu", weights_only=False)
                    if not bool(lbl["contact_pair"].any()):
                        skipped.append(f"{pair}/{obj}/{ep.stem}")
                        continue
                    cut = min(peak_end(lbl["norm_a"], args.eps),
                              peak_end(lbl["norm_b"], args.eps))
                    total_orig_frames += len(lbl["frames"])
                    total_new_frames += cut + 1
                    total_kept += 1
                    continue
                n_orig, n_new, kept = process_episode(
                    pair, obj, ep,
                    label_out_root,
                    latent_in_root, latent_out_root,
                    args.eps, args.link_mode,
                )
                total_orig_frames += n_orig
                total_new_frames += n_new
                if kept:
                    total_kept += 1
                else:
                    skipped.append(f"{pair}/{obj}/{ep.stem}")

    print(f"\nProcessed {total_eps} episodes ({total_kept} kept, {len(skipped)} skipped)")
    if total_orig_frames:
        ratio = total_new_frames / total_orig_frames
        print(f"Frames: {total_orig_frames} -> {total_new_frames} ({ratio:.1%})")
    if skipped:
        print(f"Skipped (no contact_pair=1 anywhere): {len(skipped)}")
        for s in skipped[:10]:
            print(f"  {s}")
        if len(skipped) > 10:
            print(f"  ... +{len(skipped) - 10} more")


if __name__ == "__main__":
    main()
