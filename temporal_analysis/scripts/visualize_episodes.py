"""
Visualize processed cross-modal contact data as videos.

Creates side-by-side videos: Daimon depth image (left) + eFlesh magnetic bar chart (right)
for 2 random episodes from each object.

Usage:
  python scripts/visualize_episodes.py
  python scripts/visualize_episodes.py --objects cylinder large_ball --eps_per_object 3
  python scripts/visualize_episodes.py --output_dir visualizations/
"""

import os
import sys
import glob
import argparse
import random

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize processed cross-modal episodes")
    parser.add_argument("--data_dir", type=str, default="processed_data/cross_modal",
                        help="Processed data root directory")
    parser.add_argument("--objects", nargs="+", default=None,
                        help="Object names to visualize. Default: all objects found.")
    parser.add_argument("--eps_per_object", type=int, default=2,
                        help="Number of random episodes per object")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_episodes(daimon_obj_dir):
    """Get dict of {episode_id: sorted list of .pt file paths}."""
    all_files = sorted(glob.glob(os.path.join(daimon_obj_dir, "*.pt")))
    episodes = {}
    for f in all_files:
        ep_id = os.path.basename(f).split("_fr")[0]
        episodes.setdefault(ep_id, []).append(f)
    return episodes


def load_frame(daimon_path, eflesh_dir):
    """Load a single paired frame."""
    fname = os.path.basename(daimon_path)
    eflesh_path = os.path.join(eflesh_dir, fname)

    d_data = torch.load(daimon_path, map_location="cpu", weights_only=True)
    e_data = torch.load(eflesh_path, map_location="cpu", weights_only=True)

    depth = d_data["image"].squeeze().numpy()      # (240, 320)
    magnetic = e_data["magnetic"].numpy()           # (15,)
    return depth, magnetic


MAGNET_LABELS = [
    "M0_x", "M0_y", "M0_z",
    "M1_x", "M1_y", "M1_z",
    "M2_x", "M2_y", "M2_z",
    "M3_x", "M3_y", "M3_z",
    "M4_x", "M4_y", "M4_z",
]


def make_episode_video(daimon_files, eflesh_dir, obj_name, ep_id, output_path, fps):
    """Create a video for one episode: depth image + magnetic bar chart."""
    # Pre-load all frames to get global ranges
    frames = []
    for f in tqdm(daimon_files, desc=f"  Loading {obj_name}/{ep_id}", leave=False):
        depth, magnetic = load_frame(f, eflesh_dir)
        frames.append((depth, magnetic))

    all_mag = np.stack([f[1] for f in frames], axis=0)
    mag_min = all_mag.min()
    mag_max = all_mag.max()
    mag_abs_max = max(abs(mag_min), abs(mag_max)) * 1.1

    all_depth = np.stack([f[0] for f in frames], axis=0)
    d_vmin = np.percentile(all_depth, 1)
    d_vmax = np.percentile(all_depth, 99)

    # Set up figure
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle(f"{obj_name} — {ep_id} ({len(frames)} contact frames)", fontsize=14)

    # Initial frame
    im = ax_img.imshow(frames[0][0], cmap="gray", vmin=d_vmin, vmax=d_vmax)
    ax_img.set_title("Daimon Depth", fontsize=12)
    ax_img.axis("off")

    bar_colors = []
    for i in range(5):
        bar_colors.extend([f"C{i}"] * 3)
    bars = ax_bar.barh(range(15), frames[0][1], color=bar_colors)
    ax_bar.set_xlim(-mag_abs_max, mag_abs_max)
    ax_bar.set_yticks(range(15))
    ax_bar.set_yticklabels(MAGNET_LABELS, fontsize=8)
    ax_bar.set_title("eFlesh Magnetic (15D)", fontsize=12)
    ax_bar.axvline(x=0, color="gray", linewidth=0.5)
    ax_bar.invert_yaxis()

    frame_text = ax_img.text(5, 15, "", color="white", fontsize=10,
                             bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))

    def update(i):
        depth, magnetic = frames[i]
        im.set_data(depth)
        for bar, val in zip(bars, magnetic):
            bar.set_width(val)
        frame_text.set_text(f"Frame {i+1}/{len(frames)}")
        return [im] + list(bars) + [frame_text]

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 // fps, blit=True)
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved: {output_path} ({len(frames)} frames)")


def main():
    args = parse_args()
    random.seed(args.seed)

    daimon_root = os.path.join(args.data_dir, "daimon")
    eflesh_root = os.path.join(args.data_dir, "eflesh")

    # Discover objects
    if args.objects is None:
        all_dirs = sorted(os.listdir(daimon_root))
        args.objects = [d for d in all_dirs if not d.endswith("_val")]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Cross-Modal Episode Visualization")
    print("=" * 60)
    print(f"  Objects        : {args.objects}")
    print(f"  Eps per object : {args.eps_per_object}")
    print(f"  FPS            : {args.fps}")
    print(f"  Output         : {args.output_dir}")
    print()

    for obj_name in args.objects:
        daimon_obj_dir = os.path.join(daimon_root, obj_name)
        eflesh_obj_dir = os.path.join(eflesh_root, obj_name)

        if not os.path.isdir(daimon_obj_dir):
            print(f"[{obj_name}] SKIPPED — not found")
            continue

        episodes = get_episodes(daimon_obj_dir)
        ep_ids = sorted(episodes.keys())
        print(f"[{obj_name}] {len(ep_ids)} episodes, picking {args.eps_per_object} random")

        selected = random.sample(ep_ids, min(args.eps_per_object, len(ep_ids)))

        for ep_id in selected:
            output_path = os.path.join(args.output_dir, f"{obj_name}_{ep_id}.gif")
            make_episode_video(
                episodes[ep_id], eflesh_obj_dir,
                obj_name, ep_id, output_path, args.fps
            )

    print()
    print("Done!")


if __name__ == "__main__":
    main()
