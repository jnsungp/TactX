"""
Scan processed_data/cross_modal/ to auto-detect all objects,
compute combined normalization stats from training tools,
and update config/datasets_config.yaml in-place.

Usage:
  python scripts/update_cross_modal_stats.py
  python scripts/update_cross_modal_stats.py --dry_run   # stats만 출력, yaml 수정 안 함
"""

import os
import glob
import argparse
import re

import torch
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-update cross_modal stats in datasets_config.yaml")
    parser.add_argument("--daimon_path", type=str, default="processed_data/cross_modal/daimon")
    parser.add_argument("--eflesh_path", type=str, default="processed_data/cross_modal/eflesh")
    parser.add_argument("--config_path", type=str, default="config/datasets_config.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Print stats without modifying yaml")
    parser.add_argument("--train", nargs="+", default=None,
                        help="Training tool names (e.g. --train large_ball cylinder). "
                             "If not set, auto-detects all non-val folders.")
    parser.add_argument("--val", nargs="+", default=None,
                        help="Val tool names (e.g. --val large_ball_val cylinder_val). "
                             "If not set, auto-detects all *_val folders.")
    return parser.parse_args()


def discover_tools(daimon_path):
    """Auto-detect training and val tool names from directory structure."""
    all_dirs = sorted([
        d for d in os.listdir(daimon_path)
        if os.path.isdir(os.path.join(daimon_path, d))
    ])
    val_tools = sorted([d for d in all_dirs if d.endswith("_val")])
    training_tools = sorted([d for d in all_dirs if not d.endswith("_val")])
    return training_tools, val_tools


def compute_stats(daimon_path, eflesh_path, training_tools):
    """Compute combined daimon/eflesh stats from all training tools."""
    d_sum = 0.0
    d_sq_sum = 0.0
    d_count = 0
    all_eflesh = []
    tool_counts = {}

    for tool in training_tools:
        d_files = sorted(glob.glob(os.path.join(daimon_path, tool, "*.pt")))
        e_files = sorted(glob.glob(os.path.join(eflesh_path, tool, "*.pt")))
        assert len(d_files) == len(e_files), \
            f"Count mismatch for {tool}: daimon={len(d_files)}, eflesh={len(e_files)}"
        tool_counts[tool] = len(d_files)

        for d_path, e_path in tqdm(zip(d_files, e_files), total=len(d_files), desc=tool):
            d_data = torch.load(d_path, map_location="cpu", weights_only=True)
            img = d_data["image"].float().numpy()
            d_sum += float(img.sum())
            d_sq_sum += float((img ** 2).sum())
            d_count += img.size

            e_data = torch.load(e_path, map_location="cpu", weights_only=True)
            mag = e_data["magnetic"].float().numpy()
            all_eflesh.append(mag)

    d_mean = d_sum / d_count
    d_std = float(np.sqrt(d_sq_sum / d_count - d_mean ** 2))

    eflesh_arr = np.stack(all_eflesh, axis=0)
    e_mean = eflesh_arr.mean(axis=0).tolist()
    e_std = eflesh_arr.std(axis=0).tolist()

    return d_mean, d_std, e_mean, e_std, tool_counts


def update_yaml(config_path, training_tools, val_tools, d_mean, d_std, e_mean, e_std, tool_counts):
    """Rewrite the dataset_cross_modal block in datasets_config.yaml."""
    with open(config_path, "r") as f:
        content = f.read()

    total_frames = sum(tool_counts.values())
    tool_summary = ", ".join(f"{t} ({tool_counts[t]})" for t in training_tools)

    new_block = f"""# ============================================================
# Cross-Modal Dataset: Daimon (depth from s0) + eFlesh (magnetic)
# {len(training_tools)} objects: {tool_summary} = {total_frames} frames
# Contact filter: both sensors must be in contact
# Stats computed from all {len(training_tools)} training objects combined
# ============================================================
dataset_cross_modal:
  dataset_type: 'cross_modal'

  daimon_transforms:
    - 'resize'
    - 'normalize'

  eflesh_transforms:
    - 'normalize'

  image_size: 224
  daimon_channels: 3

  daimon_stats:
    mean: [{d_mean:.6f}, {d_mean:.6f}, {d_mean:.6f}]
    std: [{d_std:.6f}, {d_std:.6f}, {d_std:.6f}]

  eflesh_stats:
    mean: [{', '.join(f'{x:.6f}' for x in e_mean)}]
    std: [{', '.join(f'{x:.6f}' for x in e_std)}]

  daimon_path: 'processed_data/cross_modal/daimon'
  eflesh_path: 'processed_data/cross_modal/eflesh'

  n_classes_training: {len(training_tools)}
  n_classes_test: 0

  tools:
    training_tools:
"""
    for t in training_tools:
        new_block += f"      - '{t}'\n"

    new_block += "\n    val_tools:\n"
    for t in val_tools:
        new_block += f"      - '{t}'\n"

    new_block += "\n    test_tools: []"

    pattern = re.compile(
        r"# =+\n# Cross-Modal Dataset.*?test_tools: \[\]",
        re.DOTALL
    )

    if pattern.search(content):
        new_content = pattern.sub(new_block, content)
    else:
        new_content = content + "\n\n" + new_block

    with open(config_path, "w") as f:
        f.write(new_content)


def main():
    args = parse_args()

    print("Scanning processed data...")
    auto_train, auto_val = discover_tools(args.daimon_path)

    training_tools = args.train if args.train is not None else auto_train
    val_tools = args.val if args.val is not None else auto_val

    # validate that specified folders exist
    for t in training_tools + val_tools:
        d = os.path.join(args.daimon_path, t)
        e = os.path.join(args.eflesh_path, t)
        assert os.path.isdir(d), f"Daimon folder not found: {d}"
        assert os.path.isdir(e), f"eFlesh folder not found: {e}"

    print(f"  Training tools: {training_tools}")
    print(f"  Val tools:      {val_tools}")
    print()

    print("Computing stats from training data...")
    d_mean, d_std, e_mean, e_std, tool_counts = compute_stats(
        args.daimon_path, args.eflesh_path, training_tools
    )

    total = sum(tool_counts.values())
    print(f"\n{'=' * 60}")
    print(f"Objects: {len(training_tools)}, Total frames: {total}")
    for t, c in tool_counts.items():
        print(f"  {t}: {c}")
    print(f"\ndaimon_mean: {d_mean:.6f}")
    print(f"daimon_std:  {d_std:.6f}")
    print(f"eflesh_mean: {[round(x, 6) for x in e_mean]}")
    print(f"eflesh_std:  {[round(x, 6) for x in e_std]}")
    print(f"{'=' * 60}")

    if args.dry_run:
        print("\n[DRY RUN] datasets_config.yaml was NOT modified.")
    else:
        update_yaml(args.config_path, training_tools, val_tools,
                     d_mean, d_std, e_mean, e_std, tool_counts)
        print(f"\nUpdated: {args.config_path}")


if __name__ == "__main__":
    main()
