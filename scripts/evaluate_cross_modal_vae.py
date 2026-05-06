"""
Qualitative evaluation of a trained CrossModalVAE checkpoint.

Produces per-object:
  - Static grid PNG:  Daimon input | self-recon D_img(z_i) | cross-recon D_img(z_j)
  - eFlesh bar PNG:   eFlesh input | self-recon D_mag(z_j) | cross-recon D_mag(z_i)
  - Per-object L1 metrics (self vs cross, both modalities)

With --episode_viz:
  - Per-episode strip PNG: 3 rows × N-frames columns for image side
  - Per-episode heatmap PNG: time × taxel for magnetic side
  - Per-episode MP4/GIF (--video): 2×3 animated grid at --fps

Usage:
  python scripts/evaluate_cross_modal_vae.py \\
      --checkpoint_dir checkpoints/cross_modal/cross_modal_vae/dataset_cross_modal_run_20260420_223654 \\
      --device cuda:0 --out_dir eval_results/cross_modal_vae \\
      --episode_viz --video

  # force a specific object / episode:
  python scripts/evaluate_cross_modal_vae.py ... --episode_obj can_val --episode_id ep0002

  
python scripts/evaluate_cross_modal_vae.py \
  --checkpoint_dir checkpoints/cross_modal/cross_modal_vae/dataset_cross_modal_run_20260420_223654 \
  --device cuda:0 \
  --objects large_ball_val \
  --episode_obj large_ball_val --episode_id ep0002 \
  --episode_viz --video
"""

from __future__ import annotations

import argparse
import os
import re
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.utils.models_setup import get_model
from joint_embedding_learning.utils.utils import get_config_details

_EP_FR_RE = re.compile(r"(ep\d+)_fr(\d+)\.pt$")


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_dir: str, device: torch.device):
    """Load CrossModalVAE from a run directory (model.pt + model_details.yaml)."""
    model_details_fp = os.path.join(checkpoint_dir, "model_details.yaml")
    model_pt_fp = os.path.join(checkpoint_dir, "model.pt")
    with open(model_details_fp) as f:
        model_details = yaml.safe_load(f)
    model = get_model("cross_modal_vae", model_details)
    state = torch.load(model_pt_fp, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, model_details


def build_transforms(dataset_details: dict):
    """Return (daimon_transform, eflesh_mean, eflesh_std)."""
    img_size = dataset_details.get("image_size", 224)
    daimon_stats = dataset_details.get("daimon_stats", {})
    mean = daimon_stats.get("mean", [0.0, 0.0, 0.0])
    std = daimon_stats.get("std", [1.0, 1.0, 1.0])
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(torch.tensor(mean), torch.tensor(std)),
    ])
    eflesh_stats = dataset_details.get("eflesh_stats", {})
    e_mean = torch.tensor(eflesh_stats.get("mean", [0.0] * 15), dtype=torch.float32)
    e_std  = torch.tensor(eflesh_stats.get("std",  [1.0] * 15), dtype=torch.float32)
    return tf, e_mean, e_std


def load_frame(daimon_fp: str, eflesh_fp: str,
               daimon_tf, e_mean: torch.Tensor, e_std: torch.Tensor,
               daimon_channels: int = 3):
    """Load one paired frame and apply train-time normalisation. Returns (x_i, x_j)."""
    d = torch.load(daimon_fp, weights_only=False)
    e = torch.load(eflesh_fp, weights_only=False)
    img = d["image"].float()
    if img.dim() == 2:
        img = img.unsqueeze(0)
    if daimon_channels == 3 and img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = daimon_tf(img)
    mag = e["magnetic"].float()
    mag = (mag - e_mean) / (e_std + 1e-8)
    return img, mag


# ──────────────────────────────────────────────────────────────────────────────
# Image de-normalisation for display
# ──────────────────────────────────────────────────────────────────────────────

def denorm_image(t: torch.Tensor,
                 mean=(0.126311, 0.126311, 0.126311),
                 std=(0.324535, 0.324535, 0.324535)) -> np.ndarray:
    """(C, H, W) → (H, W) grayscale numpy float clipped to [0, 1]."""
    m = torch.tensor(mean, dtype=t.dtype).view(-1, 1, 1)
    s = torch.tensor(std,  dtype=t.dtype).view(-1, 1, 1)
    img = (t * s + m).mean(0)   # average channels (all identical for depth)
    return img.clamp(0, 1).numpy()


def denorm_recon_image(t: torch.Tensor) -> np.ndarray:
    """Raw network output (C, H, W) → (H, W) clipped to [0, 1] for display."""
    return t.mean(0).clamp(0, 1).numpy()


def mag_to_display(v: np.ndarray, rows: int = 5, cols: int = 3) -> np.ndarray:
    """Reshape a 15-dim normalised magnetic vector to (rows, cols) for imshow.
    Rescale from roughly [-3, 3] normalised values to [0, 1] display range."""
    patch = v[:rows * cols].reshape(rows, cols)
    mn, mx = patch.min(), patch.max()
    if mx - mn > 1e-6:
        patch = (patch - mn) / (mx - mn)
    else:
        patch = np.clip(patch, 0, 1)
    return patch


# ──────────────────────────────────────────────────────────────────────────────
# Episode discovery
# ──────────────────────────────────────────────────────────────────────────────

def list_episodes(daimon_dir: str, eflesh_dir: str, obj: str):
    """Return {ep_id: [(frame_idx, daimon_fp, eflesh_fp), ...]} for *obj* folder."""
    d_path = os.path.join(daimon_dir, obj)
    e_path = os.path.join(eflesh_dir, obj)
    if not (os.path.isdir(d_path) and os.path.isdir(e_path)):
        return {}
    shared = set(os.listdir(d_path)) & set(os.listdir(e_path))
    eps: dict[str, list] = {}
    for name in shared:
        m = _EP_FR_RE.match(name)
        if m:
            fr = int(m.group(2))
            eps.setdefault(m.group(1), []).append(
                (fr, os.path.join(d_path, name), os.path.join(e_path, name))
            )
    return {ep: sorted(frames, key=lambda x: x[0]) for ep, frames in eps.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Static qualitative grid  (N random frames)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_image_grid(model, frames, daimon_tf, e_mean, e_std,
                    device, out_path: str, n_viz: int = 8):
    """
    3-row × n_viz-col PNG for the Daimon side:
        Row 0: Input Daimon
        Row 1: Self-recon  D_img(z_i)
        Row 2: Cross-recon D_img(z_j)  [magnetic z decodes to image]
    """
    sel = random.sample(frames, min(n_viz, len(frames)))
    x_i_list, x_j_list = [], []
    for _, dfp, efp in sel:
        xi, xj = load_frame(dfp, efp, daimon_tf, e_mean, e_std)
        x_i_list.append(xi); x_j_list.append(xj)
    x_i = torch.stack(x_i_list).to(device)
    x_j = torch.stack(x_j_list).to(device)

    out = model(x_i, x_j)
    n = len(sel)
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 6.6))
    row_titles = ["Input Daimon", "Self-recon  D_img(z_i)", "Cross-recon  D_img(z_j)"]
    for j in range(n):
        imgs = [
            denorm_image(x_i[j].cpu()),
            denorm_recon_image(out["recon_ii"][j].cpu()),
            denorm_recon_image(out["recon_ji"][j].cpu()),
        ]
        for i, img in enumerate(imgs):
            ax = axes[i, j] if n > 1 else axes[i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_titles[i], fontsize=9)
    plt.suptitle("Daimon reconstructions from z", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


@torch.no_grad()
def save_magnetic_grid(model, frames, daimon_tf, e_mean, e_std,
                       device, out_path: str, n_viz: int = 8):
    """
    3-row × n_viz-col PNG for the eFlesh side (reshaped 5×3 patches):
        Row 0: Input eFlesh
        Row 1: Self-recon  D_mag(z_j)
        Row 2: Cross-recon D_mag(z_i)  [image z decodes to magnetic]
    """
    sel = random.sample(frames, min(n_viz, len(frames)))
    x_i_list, x_j_list = [], []
    for _, dfp, efp in sel:
        xi, xj = load_frame(dfp, efp, daimon_tf, e_mean, e_std)
        x_i_list.append(xi); x_j_list.append(xj)
    x_i = torch.stack(x_i_list).to(device)
    x_j = torch.stack(x_j_list).to(device)

    out = model(x_i, x_j)
    n = len(sel)
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 4.5))
    row_titles = ["Input eFlesh", "Self-recon  D_mag(z_j)", "Cross-recon  D_mag(z_i)"]
    for j in range(n):
        vecs = [
            x_j[j].cpu().numpy(),
            out["recon_jj"][j].cpu().numpy(),
            out["recon_ij"][j].cpu().numpy(),
        ]
        for i, v in enumerate(vecs):
            ax = axes[i, j] if n > 1 else axes[i]
            ax.imshow(mag_to_display(v), cmap="viridis", vmin=0, vmax=1,
                      aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_titles[i], fontsize=9)
    plt.suptitle("eFlesh reconstructions from z  (5×3 taxels)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-object L1 metrics
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_object(model, all_frames, daimon_tf, e_mean, e_std,
                device, batch_size=64):
    """Return dict of averaged L1 recon losses (ii, ji, jj, ij) for all frames."""
    sums = {"ii": 0.0, "ji": 0.0, "jj": 0.0, "ij": 0.0, "n": 0}
    for start in range(0, len(all_frames), batch_size):
        batch = all_frames[start:start + batch_size]
        x_i_list, x_j_list = [], []
        for _, dfp, efp in batch:
            xi, xj = load_frame(dfp, efp, daimon_tf, e_mean, e_std)
            x_i_list.append(xi); x_j_list.append(xj)
        x_i = torch.stack(x_i_list).to(device)
        x_j = torch.stack(x_j_list).to(device)
        out = model(x_i, x_j)
        B = x_i.shape[0]
        sums["ii"] += B * F.l1_loss(out["recon_ii"], x_i).item()
        sums["ji"] += B * F.l1_loss(out["recon_ji"], x_i).item()
        sums["jj"] += B * F.l1_loss(out["recon_jj"], x_j).item()
        sums["ij"] += B * F.l1_loss(out["recon_ij"], x_j).item()
        sums["n"] += B
    n = max(sums["n"], 1)
    return {k: sums[k] / n for k in ("ii", "ji", "jj", "ij")}


# ──────────────────────────────────────────────────────────────────────────────
# Episode strip (PNG)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_episode_strip(model, ep_frames, daimon_tf, e_mean, e_std,
                       device, obj: str, ep_id: str, out_dir: str):
    """
    Two PNGs for one full episode:
      image_strip_<obj>_<ep>.png:   3 rows × N cols (input | self | cross)
      mag_heatmap_<obj>_<ep>.png:   3 stacked time×taxel heatmaps
    """
    x_i_list, x_j_list = [], []
    for _, dfp, efp in ep_frames:
        xi, xj = load_frame(dfp, efp, daimon_tf, e_mean, e_std)
        x_i_list.append(xi); x_j_list.append(xj)
    x_i = torch.stack(x_i_list).to(device)
    x_j = torch.stack(x_j_list).to(device)

    out = model(x_i, x_j)
    N = x_i.shape[0]

    # ── Image strip ──
    inp_L  = np.stack([denorm_image(x_i[k].cpu())       for k in range(N)])
    rec_ii = np.stack([denorm_recon_image(out["recon_ii"][k].cpu()) for k in range(N)])
    rec_ji = np.stack([denorm_recon_image(out["recon_ji"][k].cpu()) for k in range(N)])

    fig, axes = plt.subplots(3, N, figsize=(1.4 * N, 4.2))
    row_titles = ["Input Daimon", "Self  D_img(z_i)", "Cross  D_img(z_j)"]
    for j in range(N):
        for i, img in enumerate([inp_L[j], rec_ii[j], rec_ji[j]]):
            ax = axes[i, j] if N > 1 else axes[i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_titles[i], fontsize=9)
            if i == 0:
                ax.set_title(f"fr{ep_frames[j][0]:04d}", fontsize=7)
    plt.suptitle(f"Image recon — {obj}/{ep_id}  ({N} frames)", fontsize=11)
    plt.tight_layout()
    strip_path = os.path.join(out_dir, f"image_strip_{obj}_{ep_id}.png")
    plt.savefig(strip_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {strip_path}")

    # ── Magnetic heatmap ──
    inp_R  = x_j.cpu().numpy().T                      # (15, N)
    rec_jj = out["recon_jj"].cpu().numpy().T          # (15, N)
    rec_ij = out["recon_ij"].cpu().numpy().T          # (15, N)

    fig, axes = plt.subplots(3, 1, figsize=(max(6, 0.25 * N), 6), sharex=True)
    for ax, data, title in zip(
        axes,
        [inp_R, rec_jj, rec_ij],
        ["Input eFlesh (normalised)",
         "Self-recon  D_mag(z_j)",
         "Cross-recon  D_mag(z_i)"],
    ):
        im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_ylabel("taxel"); ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.02)
    axes[-1].set_xlabel("frame")
    plt.suptitle(f"Magnetic recon — {obj}/{ep_id}  ({N} frames)", fontsize=11)
    mag_path = os.path.join(out_dir, f"mag_heatmap_{obj}_{ep_id}.png")
    plt.savefig(mag_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {mag_path}")

    # Recon diagnostics
    l_ii = F.l1_loss(out["recon_ii"], x_i).item()
    l_ji = F.l1_loss(out["recon_ji"], x_i).item()
    l_jj = F.l1_loss(out["recon_jj"], x_j).item()
    l_ij = F.l1_loss(out["recon_ij"], x_j).item()
    print(f"  {ep_id} {obj}: {N} frames  "
          f"img-self={l_ii:.4f} img-cross={l_ji:.4f} | "
          f"mag-self={l_jj:.4f} mag-cross={l_ij:.4f}")

    return inp_L, rec_ii, rec_ji, inp_R, rec_jj, rec_ij


# ──────────────────────────────────────────────────────────────────────────────
# Episode MP4 / GIF
# ──────────────────────────────────────────────────────────────────────────────

def save_episode_video(inp_L, rec_ii, rec_ji,
                       inp_R, rec_jj, rec_ij,
                       ep_frames, obj: str, ep_id: str,
                       out_stem: str, fps: int) -> str:
    """
    2×3 animated grid:
      Row 0 (image):    Input Daimon | Self D_img(z_i) | Cross D_img(z_j)
      Row 1 (magnetic): Input eFlesh | Self D_mag(z_j) | Cross D_mag(z_i)

    eFlesh 15-d vector reshaped to 5×3 for imshow.
    Tries mp4 (ffmpeg), falls back to gif (Pillow).
    """
    N = inp_L.shape[0]
    MAG_R, MAG_C = 5, 3

    # Pre-process magnetic: normalised values to [0,1] per column for display
    def prep_mag(mat_15xN):
        # mat_15xN: (15, N)  → list of (5, 3) arrays
        frames_out = []
        for t in range(N):
            v = mat_15xN[:, t]
            patch = v.reshape(MAG_R, MAG_C)
            mn, mx = mat_15xN.min(), mat_15xN.max()
            if mx - mn > 1e-6:
                patch = (patch - mn) / (mx - mn)
            frames_out.append(patch.clip(0, 1))
        return frames_out

    mag_inp  = prep_mag(inp_R)
    mag_jj   = prep_mag(rec_jj)
    mag_ij   = prep_mag(rec_ij)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    titles = [
        ["Input Daimon", "Self  D_img(z_i)", "Cross  D_img(z_j)"],
        ["Input eFlesh (5×3)", "Self  D_mag(z_j)", "Cross  D_mag(z_i)"],
    ]
    ims = [[None] * 3, [None] * 3]
    blank_img = np.zeros_like(inp_L[0])
    blank_mag = np.zeros((MAG_R, MAG_C))
    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            ax.set_title(titles[r][c], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ims[r][c] = ax.imshow(blank_img, cmap="gray", vmin=0, vmax=1,
                                      interpolation="bilinear")
            else:
                ims[r][c] = ax.imshow(blank_mag, cmap="viridis", vmin=0, vmax=1,
                                      interpolation="nearest")
    suptitle = fig.suptitle("", fontsize=11)
    plt.tight_layout()

    def update(i: int):
        ims[0][0].set_data(inp_L[i])
        ims[0][1].set_data(rec_ii[i])
        ims[0][2].set_data(rec_ji[i])
        ims[1][0].set_data(mag_inp[i])
        ims[1][1].set_data(mag_jj[i])
        ims[1][2].set_data(mag_ij[i])
        fr_idx = ep_frames[i][0]
        suptitle.set_text(f"{obj}/{ep_id}  frame {fr_idx:04d}  ({i+1}/{N})")
        return [ims[0][0], ims[0][1], ims[0][2],
                ims[1][0], ims[1][1], ims[1][2], suptitle]

    anim = FuncAnimation(fig, update, frames=N, interval=1000 / fps, blit=False)

    mp4_path = out_stem + ".mp4"
    try:
        anim.save(mp4_path, writer=FFMpegWriter(fps=fps, bitrate=1800))
        plt.close(fig)
        print(f"  saved {mp4_path}")
        return mp4_path
    except Exception as exc:
        print(f"  ffmpeg unavailable ({exc.__class__.__name__}); saving gif instead")
        gif_path = out_stem + ".gif"
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  saved {gif_path}")
        return gif_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CrossModalVAE reconstruction quality")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Run directory with model.pt + model_details.yaml")
    p.add_argument("--dataset_config", type=str,
                   default="config/datasets_config.yaml")
    p.add_argument("--dataset_name", type=str, default="dataset_cross_modal")
    p.add_argument("--objects", type=str, nargs="+", default=None,
                   help="Override which objects to evaluate (default: val_tools from config)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="eval_results/cross_modal_vae")
    p.add_argument("--n_viz", type=int, default=8,
                   help="Random frames per static grid")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode_viz", action="store_true",
                   help="Save per-episode strip PNGs (and optionally video)")
    p.add_argument("--episode_obj", type=str, default=None,
                   help="Object for episode viz, e.g. can_val (default: random)")
    p.add_argument("--episode_id", type=str, default=None,
                   help="Specific episode like ep0002 (default: random)")
    p.add_argument("--video", action="store_true",
                   help="Also export an mp4/gif animation per episode")
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model ──
    ckpt_dir = args.checkpoint_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                ckpt_dir)
    model, model_details = load_checkpoint(ckpt_dir, device)
    print("=" * 60)
    print("CrossModalVAE Reconstruction Evaluation")
    print("=" * 60)
    print(f"  Checkpoint  : {ckpt_dir}")
    print(f"  projection_dim={model_details['projection_dim']}  "
          f"resnet={model_details['resnet']}")
    print(f"  Device      : {device}")
    print(f"  Output dir  : {args.out_dir}")

    # ── Dataset config ──
    ds_cfg_fp = args.dataset_config
    if not os.path.isabs(ds_cfg_fp):
        ds_cfg_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 ds_cfg_fp)
    dataset_details = get_config_details(ds_cfg_fp, args.dataset_name)
    daimon_tf, e_mean, e_std = build_transforms(dataset_details)
    daimon_channels = dataset_details.get("daimon_channels", 3)

    daimon_base = dataset_details["daimon_path"]
    eflesh_base = dataset_details["eflesh_path"]
    if not os.path.isabs(daimon_base):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        daimon_base = os.path.join(root, daimon_base)
        eflesh_base = os.path.join(root, eflesh_base)

    objects = args.objects or dataset_details["tools"].get("val_tools", [])
    print(f"  Objects     : {objects}\n")

    # ── Per-object loop ──
    agg = {"ii": 0.0, "ji": 0.0, "jj": 0.0, "ij": 0.0, "n": 0}
    for obj in objects:
        d_dir = os.path.join(daimon_base, obj)
        e_dir = os.path.join(eflesh_base, obj)
        if not os.path.isdir(d_dir):
            print(f"[skip] {obj}: directory not found ({d_dir})")
            continue

        # Collect all frames across all episodes
        all_frames = []
        for name in sorted(os.listdir(d_dir)):
            if _EP_FR_RE.match(name):
                all_frames.append(
                    (int(_EP_FR_RE.match(name).group(2)),
                     os.path.join(d_dir, name),
                     os.path.join(e_dir, name))
                )
        if not all_frames:
            print(f"[skip] {obj}: no .pt frames found")
            continue

        print(f"[{obj}]  {len(all_frames)} frames")

        # Metrics
        m = eval_object(model, all_frames, daimon_tf, e_mean, e_std,
                        device, batch_size=args.batch_size)
        print(f"   L1  img-self={m['ii']:.4f}  img-cross={m['ji']:.4f} | "
              f"mag-self={m['jj']:.4f}  mag-cross={m['ij']:.4f}")
        for k in ("ii", "ji", "jj", "ij"):
            agg[k] += len(all_frames) * m[k]
        agg["n"] += len(all_frames)

        # Static grids
        save_image_grid(
            model, all_frames, daimon_tf, e_mean, e_std, device,
            os.path.join(args.out_dir, f"grid_image_{obj}.png"),
            n_viz=args.n_viz,
        )
        save_magnetic_grid(
            model, all_frames, daimon_tf, e_mean, e_std, device,
            os.path.join(args.out_dir, f"grid_magnetic_{obj}.png"),
            n_viz=args.n_viz,
        )

    # ── Aggregate metrics ──
    if agg["n"] > 0:
        n = agg["n"]
        print("\n" + "=" * 60)
        print(f"AGGREGATE  ({n} val frames, {len(objects)} objects)")
        print(f"   img-self={agg['ii']/n:.4f}  img-cross={agg['ji']/n:.4f} | "
              f"mag-self={agg['jj']/n:.4f}  mag-cross={agg['ij']/n:.4f}")
        if agg["ii"] / n < agg["ji"] / n:
            print("   ✓  self < cross  (expected: model learns from own modality first)")
        else:
            print("   ✗  cross ≤ self  (training may need more epochs / lambda_recon tuning)")

    # ── Episode visualization ──
    if args.episode_viz:
        rng = random.Random(args.seed)
        ep_obj = args.episode_obj or rng.choice(objects)
        eps = list_episodes(daimon_base, eflesh_base, ep_obj)
        if not eps:
            print(f"\n[episode_viz] no episodes found for {ep_obj}"); return
        ep_id = args.episode_id or rng.choice(sorted(eps.keys()))
        if ep_id not in eps:
            print(f"\n[episode_viz] episode {ep_id} not found in {ep_obj}; "
                  f"available: {sorted(eps.keys())[:10]}"); return

        ep_frames = eps[ep_id]
        print(f"\n── Episode viz: {ep_obj}/{ep_id}  ({len(ep_frames)} frames) ──")
        inp_L, rec_ii, rec_ji, inp_R, rec_jj, rec_ij = save_episode_strip(
            model, ep_frames, daimon_tf, e_mean, e_std,
            device, ep_obj, ep_id, args.out_dir,
        )

        if args.video:
            stem = os.path.join(args.out_dir, f"episode_{ep_obj}_{ep_id}")
            save_episode_video(
                inp_L, rec_ii, rec_ji,
                inp_R, rec_jj, rec_ij,
                ep_frames, ep_obj, ep_id, stem, args.fps,
            )

    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
