"""
Cross-modal data preprocessing: raw pkl episodes → .pt files for training.

Processing steps:
  1. Daimon (depth): s0에서 depth 채널 추출 (240, 320)
     - s0는 (240, 320, 5) float16으로 저장되었으나 실제 순서는
       [depth.flat, deformation.flat(2ch), shear.flat(2ch)] → depth = flat[:N]
  2. eFlesh (magnetic): s1 (5, 3) flattened to (15,)
  3. Contact filtering: 양쪽 센서 모두 접촉 상태여야 저장
     - Daimon depth mean > threshold AND eFlesh abs mean > threshold
  4. Computes normalization statistics for datasets_config.yaml

Usage:
  python scripts/preprocess_cross_modal.py
  python scripts/preprocess_cross_modal.py --contact_threshold 0.005
  python scripts/preprocess_cross_modal.py --eflesh_contact_threshold 0.01
  python scripts/preprocess_cross_modal.py --no_contact_filter
  python scripts/preprocess_cross_modal.py --no_mirror
  python scripts/preprocess_cross_modal.py --raw_train path/to/train --raw_val path/to/val --train_name my_object
"""

import os
import pickle
import shutil
import argparse

import torch
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess cross-modal tactile data")
    parser.add_argument("--raw_train", type=str,
                        default="tactile_data_new/left_daimon_right_eflesh_large_ball",
                        help="Raw training data directory")
    parser.add_argument("--raw_val", type=str,
                        default="tactile_data_new/left_daimon_right_eflesh_large_ball_val",
                        help="Raw validation data directory")
    parser.add_argument("--output_dir", type=str,
                        default="processed_data/cross_modal",
                        help="Output root directory")
    parser.add_argument("--train_name", type=str, default="large_ball",
                        help="Training tool/object name")
    parser.add_argument("--val_name", type=str, default="large_ball_val",
                        help="Validation tool/object name")
    parser.add_argument("--contact_threshold", type=float, default=0.005,
                        help="Daimon depth mean threshold for contact detection")
    parser.add_argument("--eflesh_contact_threshold", type=float, default=0.001,
                        help="eFlesh abs-mean threshold for contact detection")
    parser.add_argument("--no_contact_filter", action="store_true",
                        help="Disable contact filtering (keep all frames)")
    parser.add_argument("--no_mirror", action="store_true",
                        help="Disable horizontal mirror flip on Daimon depth")
    return parser.parse_args()


def extract_depth_from_s0(s0):
    """Extract depth channel from s0 which was wrongly reshaped during saving.

    Original flat layout: [depth(N), deformation_x(N), deformation_y(N),
                           shear_x(N), shear_y(N)] where N = 240*320.
    Saved as (240, 320, 5) by naive reshape, so we flatten and re-slice.
    """
    H, W = 240, 320
    N = H * W
    flat = s0.reshape(-1)
    return flat[:N].reshape(H, W)


def process_episodes(raw_data_root, output_daimon, output_eflesh,
                     contact_threshold, eflesh_contact_threshold,
                     mirror, label):
    """Process all episodes in a directory, saving .pt files."""
    if os.path.exists(output_daimon):
        shutil.rmtree(output_daimon)
    if os.path.exists(output_eflesh):
        shutil.rmtree(output_eflesh)
    os.makedirs(output_daimon)
    os.makedirs(output_eflesh)

    episode_dirs = sorted([
        d for d in os.listdir(raw_data_root) if d.startswith("episode_")
    ])
    num_episodes = len(episode_dirs)

    all_eflesh = []
    daimon_sum = 0.0
    daimon_sq_sum = 0.0
    total_pixels = 0
    total_saved = 0
    total_skipped = 0

    for ep in tqdm(range(num_episodes), desc=f"{label}"):
        ep_dir = os.path.join(raw_data_root, f"episode_{ep:04d}")
        if not os.path.isdir(ep_dir):
            continue

        pkl_files = sorted([
            os.path.join(ep_dir, f) for f in os.listdir(ep_dir)
            if f.endswith(".pkl") and f != "metadata.pkl"
        ])

        for frame_idx, pkl_path in enumerate(pkl_files):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            # Daimon depth: extract from s0
            s0 = data["s0"]                                # (240, 320, 5) float16
            depth = extract_depth_from_s0(s0).astype(np.float32)  # (240, 320)

            # eFlesh: (5, 3) → (15,)
            s1 = data["s1"].astype(np.float32)

            # Contact filter: BOTH sensors must be in contact
            if contact_threshold is not None:
                daimon_contact = np.abs(depth).mean() > contact_threshold
                eflesh_contact = np.abs(s1).mean() > eflesh_contact_threshold
                if not (daimon_contact and eflesh_contact):
                    total_skipped += 1
                    continue

            # Mirror flip (gripper symmetry correction)
            if mirror:
                depth = np.fliplr(depth).copy()

            daimon_tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, 240, 320)

            # Stats accumulation
            n_pix = depth.shape[0] * depth.shape[1]
            daimon_sum += depth.sum()
            daimon_sq_sum += (depth ** 2).sum()
            total_pixels += n_pix

            eflesh_tensor = torch.from_numpy(s1.flatten())
            all_eflesh.append(s1.flatten())

            fname = f"ep{ep:04d}_fr{frame_idx:04d}.pt"
            torch.save({"image": daimon_tensor}, os.path.join(output_daimon, fname))
            torch.save({"magnetic": eflesh_tensor}, os.path.join(output_eflesh, fname))
            total_saved += 1

    # Compute statistics
    stats = None
    if total_saved > 0:
        d_mean = float(daimon_sum / total_pixels)
        d_std = float(np.sqrt(daimon_sq_sum / total_pixels - d_mean ** 2))
        eflesh_all = np.stack(all_eflesh, axis=0)
        e_mean = eflesh_all.mean(axis=0).tolist()
        e_std = eflesh_all.std(axis=0).tolist()
        stats = {
            "daimon_mean": d_mean, "daimon_std": d_std,
            "eflesh_mean": e_mean, "eflesh_std": e_std,
        }

    return total_saved, total_skipped, stats


def print_yaml_config(stats):
    """Print normalization stats in datasets_config.yaml format."""
    dm = stats["daimon_mean"]
    ds = stats["daimon_std"]
    print(f"  daimon_stats:")
    print(f"    mean: [{dm:.6f}, {dm:.6f}, {dm:.6f}]")
    print(f"    std: [{ds:.6f}, {ds:.6f}, {ds:.6f}]")
    print(f"  eflesh_stats:")
    print(f"    mean: {[round(x, 6) for x in stats['eflesh_mean']]}")
    print(f"    std: {[round(x, 6) for x in stats['eflesh_std']]}")


def main():
    args = parse_args()

    mirror = not args.no_mirror
    contact_threshold = None if args.no_contact_filter else args.contact_threshold
    eflesh_contact_threshold = args.eflesh_contact_threshold

    print("=" * 60)
    print("Cross-Modal Data Preprocessing")
    print("=" * 60)
    print(f"  Daimon source  : s0 depth channel")
    print(f"  Contact filter : {'OFF' if contact_threshold is None else f'daimon={contact_threshold}, eflesh={eflesh_contact_threshold}'}")
    print(f"  Mirror flip    : {'ON' if mirror else 'OFF'}")
    print(f"  Output dir     : {args.output_dir}")
    print()

    # --- Train set ---
    if os.path.isdir(args.raw_train):
        print(f"[Train] {args.raw_train}")
        out_d = os.path.join(args.output_dir, "daimon", args.train_name)
        out_e = os.path.join(args.output_dir, "eflesh", args.train_name)
        saved, skipped, train_stats = process_episodes(
            args.raw_train, out_d, out_e, contact_threshold,
            eflesh_contact_threshold, mirror, "Train"
        )
        total = saved + skipped
        print(f"  Saved: {saved}, Skipped: {skipped} ({skipped/total*100:.1f}% non-contact)")
        print(f"  → {out_d}")
        print(f"  → {out_e}")
    else:
        print(f"[Train] SKIPPED — {args.raw_train} not found")
        train_stats = None

    print()

    # --- Val set ---
    if os.path.isdir(args.raw_val):
        print(f"[Val] {args.raw_val}")
        out_d = os.path.join(args.output_dir, "daimon", args.val_name)
        out_e = os.path.join(args.output_dir, "eflesh", args.val_name)
        saved, skipped, _ = process_episodes(
            args.raw_val, out_d, out_e, contact_threshold,
            eflesh_contact_threshold, mirror, "Val"
        )
        total = saved + skipped
        print(f"  Saved: {saved}, Skipped: {skipped} ({skipped/total*100:.1f}% non-contact)")
        print(f"  → {out_d}")
        print(f"  → {out_e}")
    else:
        print(f"[Val] SKIPPED — {args.raw_val} not found")

    # --- Print config ---
    if train_stats:
        print()
        print("=" * 60)
        print("Normalization stats (paste into datasets_config.yaml):")
        print("=" * 60)
        print_yaml_config(train_stats)


if __name__ == "__main__":
    main()
