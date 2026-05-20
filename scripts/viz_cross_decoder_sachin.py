"""Cross-decoder qualitative visualization for a trained CrossModalVAE3.

For each input sensor S in {D, E, F}, walk the frames of a pnp HDF5 episode:
    raw_S -> preprocess -> enc_S(.) -> mu_S (16-d shared latent)
    z_S -> dec_S(.)        # self reconstruction (recon of S itself)
    z_S -> dec_{other1}(.) # cross prediction in other1's space
    z_S -> dec_{other2}(.) # cross prediction in other2's space

Each video shows, frame-by-frame, the ground-truth S reading next to the
self-recon, and below it the two cross-decoded predictions in the OTHER
sensors' native spaces (image grid / 15-d bars / 12x16 heatmap).

Mismatched modalities are an intentional feature: it visualizes whether the
shared latent really captures cross-modal contact structure (zero-shot).

Usage:
    python scripts/viz_cross_decoder_sachin.py \
        --checkpoint_dir /media/ExtHDD/sachin/latent_checkpoints/dataset_sachin_run_dynalign_plain_cyl_plug_20260513_153826 \
        --output_dir /home/sachin/Workspace_cross_sensor/cross_modal_viz \
        --episodes_per_sensor 2
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from joint_embedding_learning.cross_modal.model import (
    CrossModalVAE3, MagneticEncoder, FlexiTacEncoder,
)


# ---------------------------------------------------------------------------
# Per-task per-sensor episode roots and raw shape on the HDF5 wire.
# Add new tasks by appending to TASK_TO_ROOTS keyed by task_name.
# ---------------------------------------------------------------------------
TELEOP_ROOT = '/media/ExtHDD/sachin/teleop'

TASK_TO_ROOTS = {
    'pnp': {
        'D': f'{TELEOP_ROOT}/dai_dai_object_pnp/processed',
        'E': f'{TELEOP_ROOT}/eflesh_eflesh_object_pnp/processed',
        'F': f'{TELEOP_ROOT}/flexi_flexi_object_pnp/processed',
    },
    'plug_insertion_direct': {
        'D': f'{TELEOP_ROOT}/dai_dai_plug_insertion_direct/processed',
        'E': f'{TELEOP_ROOT}/eflesh_eflesh_plug_insertion_direct/processed',
        'F': f'{TELEOP_ROOT}/flexi_flexi_plug_insertion_direct/processed',
    },
}

SENSOR_INFO = {
    'D': {'name': 'daimon',   'raw_numel': 240 * 320 * 5},
    'E': {'name': 'eflesh',   'raw_numel': 15},
    'F': {'name': 'flexitac', 'raw_numel': 12 * 16},
}

DAIMON_H, DAIMON_W = 240, 320
DAIMON_N = DAIMON_H * DAIMON_W
FLEXITAC_H, FLEXITAC_W = 12, 16


def daimon_raw_to_3ch(raw_flat: np.ndarray) -> np.ndarray:
    """raw_flat: (T, 384000). Returns (T, 3, H, W) with [depth, shear_x, shear_y].

    Matches cross_modal_tactile_workspace/scripts/preprocess_cross_modal.py
    daimon_to_chw() which selected channels 0/3/4 from the 5-channel raw.
    """
    T = raw_flat.shape[0]
    flat = raw_flat.astype(np.float32).reshape(T, 5, DAIMON_N)
    depth   = flat[:, 0]                                  # (T, N)
    shear_x = flat[:, 3]
    shear_y = flat[:, 4]
    stacked = np.stack([depth, shear_x, shear_y], axis=1)  # (T, 3, N)
    return stacked.reshape(T, 3, DAIMON_H, DAIMON_W)


# ---------------------------------------------------------------------------
# Model loader (bypasses simclr; uses torchvision resnet18 directly).
# ---------------------------------------------------------------------------

def build_vae3_from_yamls(model_details: dict) -> CrossModalVAE3:
    name = model_details['resnet']
    if name == 'resnet18':
        backbone = resnet18(weights=None)
    elif name == 'resnet50':
        backbone = resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported resnet: {name}")
    n_features = backbone.fc.in_features

    magnetic_encoder = MagneticEncoder(
        input_dim=model_details.get('magnetic_input_dim', 15),
        hidden_dims=tuple(model_details.get('magnetic_hidden_dims', [64, 128, 256])),
        output_dim=n_features,
    )
    flexi_h = model_details.get('flexitac_in_h', 12)
    flexi_w = model_details.get('flexitac_in_w', 16)
    flexitac_encoder = FlexiTacEncoder(
        in_h=flexi_h, in_w=flexi_w,
        channels=tuple(model_details.get('flexitac_encoder_channels', [32, 64])),
        pool_h=model_details.get('flexitac_pool_h', 6),
        pool_w=model_details.get('flexitac_pool_w', 8),
        output_dim=n_features,
    )
    model = CrossModalVAE3(
        image_encoder=backbone,
        magnetic_encoder=magnetic_encoder,
        flexitac_encoder=flexitac_encoder,
        projection_dim=model_details['projection_dim'],
        n_features=n_features,
        image_out_channels=model_details.get('image_out_channels', 3),
        image_out_size=model_details.get('image_out_size', 224),
        image_decoder_channels=tuple(model_details.get('image_decoder_channels',
                                                       [256, 128, 64, 32])),
        magnetic_decoder_hidden=tuple(model_details.get('magnetic_decoder_hidden',
                                                        [256, 128, 64])),
        magnetic_out_dim=model_details.get('magnetic_input_dim', 15),
        flexitac_in_h=flexi_h, flexitac_in_w=flexi_w,
        flexitac_decoder_channels=tuple(model_details.get('flexitac_decoder_channels',
                                                           [64, 32])),
        flexitac_out_channels=model_details.get('flexitac_out_channels', 1),
    )
    return model


def load_full_vae3(ckpt_dir: Path, device: str):
    with open(ckpt_dir / 'model_details.yaml') as f:
        model_details = yaml.safe_load(f)
    with open(ckpt_dir / 'dataset_details.yaml') as f:
        dataset_details = yaml.safe_load(f)
    model = build_vae3_from_yamls(model_details).to(device)
    state = torch.load(ckpt_dir / 'model.pt', map_location=device, weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f'[load_full_vae3] missing keys: {missing}')
    if unexpected:
        print(f'[load_full_vae3] unexpected keys: {unexpected}')
    model.eval()
    return model, dataset_details


# ---------------------------------------------------------------------------
# Per-sensor preprocess into model input.
# ---------------------------------------------------------------------------

def preprocess_for_encoder(sensor: str, raw_np: np.ndarray, dataset_details: dict,
                           image_size: int = 224, device: str = 'cpu') -> torch.Tensor:
    """raw_np: per-sensor numpy from HDF5 (see SENSOR_INFO[sensor]['raw_numel']).
    Returns the tensor consumed by the matching encoder branch.
    """
    if sensor == 'D':
        # (T, 384000) -> (T, 3, H, W) with channels [depth, shear_x, shear_y]
        x = daimon_raw_to_3ch(raw_np)                  # (T, 3, 240, 320)
        x = torch.from_numpy(x).float().to(device)
        x = F.interpolate(x, size=(image_size, image_size), mode='bilinear',
                          align_corners=False, antialias=True)
        stats = dataset_details['daimon_stats']
        m = torch.tensor(stats['mean'], device=device).view(1, 3, 1, 1).float()
        s = torch.tensor(stats['std'],  device=device).view(1, 3, 1, 1).float()
        return (x - m) / (s + 1e-8)

    if sensor == 'E':
        x = torch.from_numpy(raw_np).float().to(device)         # (T, 15)
        stats = dataset_details['eflesh_stats']
        m = torch.tensor(stats['mean'], device=device).view(1, -1).float()
        s = torch.tensor(stats['std'],  device=device).view(1, -1).float()
        return (x - m) / (s + 1e-8)

    if sensor == 'F':
        x = torch.from_numpy(raw_np).float().to(device).reshape(-1, FLEXITAC_H, FLEXITAC_W)
        stats = dataset_details['flexitac_stats']
        m = float(stats['mean'])
        s = float(stats['std'])
        return (x - m) / (s + 1e-8)

    raise ValueError(sensor)


# ---------------------------------------------------------------------------
# Rendering one frame for a (gt_sensor, predictions) tuple. matplotlib -> RGB.
# ---------------------------------------------------------------------------

def _norm_daimon_for_display(x: np.ndarray) -> np.ndarray:
    """x: (3, H, W) — per-channel min-max to [0, 1] for display."""
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        ch = x[c]
        lo, hi = ch.min(), ch.max()
        out[c] = (ch - lo) / (hi - lo + 1e-8)
    return out


def _ax_show_daimon(ax, x_chw: np.ndarray, title: str):
    """Show a (3, H, W) Daimon tensor as a horizontal triptych [depth, shear_x, shear_y]."""
    disp = _norm_daimon_for_display(x_chw)
    ax.imshow(np.concatenate([disp[0], disp[1], disp[2]], axis=1), cmap='viridis')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _ax_show_eflesh(ax, x_15: np.ndarray, title: str, y_lim=None):
    """Show a 15-d eFlesh vector as grouped bars (5 magnets x 3 axes)."""
    arr = x_15.reshape(5, 3)
    width = 0.25
    idx = np.arange(5)
    colors = ('tab:red', 'tab:green', 'tab:blue')
    for i, (c, label) in enumerate(zip(colors, ('x', 'y', 'z'))):
        ax.bar(idx + (i - 1) * width, arr[:, i], width, color=c, label=label)
    ax.set_xticks(idx); ax.set_xticklabels([f'm{i}' for i in range(5)], fontsize=7)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis='y', labelsize=7)
    ax.legend(fontsize=6, loc='upper right')


def _ax_show_flexi(ax, x_hw: np.ndarray, title: str, vlim=None):
    """Show a (12, 16) FlexiTac heatmap. vlim=(vmin, vmax) optional."""
    if vlim is None:
        vmin, vmax = float(np.min(x_hw)), float(np.max(x_hw))
    else:
        vmin, vmax = vlim
    ax.imshow(x_hw, cmap='magma', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


# Shape helpers — convert decoder outputs to their natural display tensors.
def _to_display(out_tensor: torch.Tensor, sensor: str) -> np.ndarray:
    """sensor: target decoder identity. Strip batch, return display-friendly numpy."""
    x = out_tensor.detach().cpu().numpy()
    if sensor == 'D':                  # (3, H, W)
        return x[0]                    # take batch 0
    if sensor == 'E':                  # (15,)
        return x[0]
    if sensor == 'F':                  # (1, 12, 16) -> (12, 16)
        return x[0, 0]
    raise ValueError(sensor)


def _gt_to_display(gt_tensor: torch.Tensor, sensor: str) -> np.ndarray:
    """Same as _to_display but operates on the encoder's input tensor."""
    x = gt_tensor.detach().cpu().numpy()
    if sensor == 'D':                  # (3, H_proc, W_proc)
        return x[0]
    if sensor == 'E':
        return x[0]
    if sensor == 'F':                  # (12, 16) or (1, 12, 16)
        if x.ndim == 3 and x.shape[1] == FLEXITAC_H:
            return x[0]
        return x[0, 0]
    raise ValueError(sensor)


# ---------------------------------------------------------------------------
# Per-episode video render.
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_episode_video(model: CrossModalVAE3,
                         dataset_details: dict,
                         sensor: str,
                         episode_path: Path,
                         out_path: Path,
                         device: str,
                         fps: int = 20,
                         max_frames: int = 400):
    info = SENSOR_INFO[sensor]
    others = [s for s in ('D', 'E', 'F') if s != sensor]

    with h5py.File(episode_path, 'r') as f:
        raw = f['observation.tac0'][:]      # use the left finger only; tac1 symmetric
    T = raw.shape[0]
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames).astype(int)
        raw = raw[idx]
        T = max_frames

    enc_in = preprocess_for_encoder(sensor, raw, dataset_details, device=device)
    # enc_in: D -> (T,3,H,W); E -> (T,15); F -> (T,12,16)

    # Encode all frames batched to keep it fast.
    BATCH = 32
    mus = []
    for i in range(0, T, BATCH):
        x = enc_in[i:i+BATCH]
        h, mu, logvar, z = model._enc(x, sensor)
        mus.append(mu)
    mu_all = torch.cat(mus, dim=0)              # (T, projection_dim)

    # Pre-roll: compute global y-axis / vmin-vmax for stable display.
    # We accumulate decoder outputs in chunks too.
    dec_outs = {s: [] for s in ('D', 'E', 'F')}
    for i in range(0, T, BATCH):
        z = mu_all[i:i+BATCH]
        for s in ('D', 'E', 'F'):
            d = model._decoder_for(s)
            dec_outs[s].append(d(z).cpu())
    for s in dec_outs:
        dec_outs[s] = torch.cat(dec_outs[s], dim=0)
    # Move GT to CPU once.
    gt_all = enc_in.cpu()
    mu_all = mu_all.cpu()

    # Display ranges for E (bars) and F (heatmap) so the video doesn't flicker.
    def _range_for(s, tensors):
        if s == 'E':
            cat = torch.cat([t.flatten() for t in tensors])
            v = max(1e-3, float(cat.abs().max().item()) * 1.1)
            return (-v, v)
        if s == 'F':
            cat = torch.cat([t.flatten() for t in tensors])
            return (float(cat.min().item()), float(cat.max().item()))
        return None

    e_pool = []
    f_pool = []
    if sensor == 'E':
        e_pool.append(gt_all)
    if sensor == 'F':
        f_pool.append(gt_all)
    e_pool.append(dec_outs['E'])
    f_pool.append(dec_outs['F'].squeeze(1) if dec_outs['F'].dim() == 4 else dec_outs['F'])
    e_range = _range_for('E', e_pool)
    f_range = _range_for('F', f_pool)

    # Figure layout: 2 rows x 2 cols (GT/self on top, two crosses below).
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    ax_gt   = fig.add_subplot(gs[0, 0])
    ax_self = fig.add_subplot(gs[0, 1])
    ax_c0   = fig.add_subplot(gs[1, 0])
    ax_c1   = fig.add_subplot(gs[1, 1])

    # First frame: render once to size the video.
    def draw(t: int):
        for ax in (ax_gt, ax_self, ax_c0, ax_c1):
            ax.clear()
        gt_np = _gt_to_display(gt_all[t:t+1], sensor)
        recon_self_np = _to_display(dec_outs[sensor][t:t+1], sensor)
        cross_a_np = _to_display(dec_outs[others[0]][t:t+1], others[0])
        cross_b_np = _to_display(dec_outs[others[1]][t:t+1], others[1])

        if sensor == 'D':
            _ax_show_daimon(ax_gt,   gt_np,        f'GT D (frame {t})')
            _ax_show_daimon(ax_self, recon_self_np, 'self: dec_D(z_D)')
        elif sensor == 'E':
            _ax_show_eflesh(ax_gt,   gt_np,        f'GT E (frame {t})', y_lim=e_range)
            _ax_show_eflesh(ax_self, recon_self_np, 'self: dec_E(z_E)',  y_lim=e_range)
        else:
            _ax_show_flexi(ax_gt,   gt_np,        f'GT F (frame {t})', vlim=f_range)
            _ax_show_flexi(ax_self, recon_self_np, 'self: dec_F(z_F)',  vlim=f_range)

        for ax, x, s in ((ax_c0, cross_a_np, others[0]),
                         (ax_c1, cross_b_np, others[1])):
            tag = f'cross: dec_{s}(z_{sensor})'
            if s == 'D':
                _ax_show_daimon(ax, x, tag)
            elif s == 'E':
                _ax_show_eflesh(ax, x, tag, y_lim=e_range)
            else:
                _ax_show_flexi(ax, x, tag, vlim=f_range)

        fig.suptitle(f'{episode_path.name}  |  encoded with sensor {sensor}',
                     fontsize=10)
        fig.canvas.draw()
        # RGB array. matplotlib >=3.8 uses buffer_rgba.
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[..., :3].copy()

    first = draw(0)
    H, W = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
    for t in range(1, T):
        frame = draw(t)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    plt.close(fig)
    print(f'  wrote {out_path}  ({T} frames, {W}x{H} @ {fps}fps)')


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', default=(
        '/media/ExtHDD/sachin/latent_checkpoints/'
        'dataset_sachin_run_dynalign_plain_cyl_plug_20260513_153826'))
    p.add_argument('--output_dir', default='/home/sachin/Workspace_cross_sensor/cross_modal_viz')
    p.add_argument('--episodes_per_sensor', type=int, default=2)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--max_frames', type=int, default=400)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--episode_indices', type=int, nargs='+', default=None,
                   help='Specific episode indices (e.g. 0 5). Defaults to first N.')
    p.add_argument('--task', default='pnp', choices=sorted(TASK_TO_ROOTS.keys()),
                   help='Task whose teleop episodes to visualize.')
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f'Loading CrossModalVAE3 from {ckpt_dir}')
    model, dataset_details = load_full_vae3(ckpt_dir, args.device)
    print(f'  projection_dim={model.projection_dim}, '
          f'image_size={dataset_details.get("image_size", 224)}')

    task_roots = TASK_TO_ROOTS[args.task]
    print(f'Task: {args.task}')

    for sensor in ('D', 'E', 'F'):
        info = SENSOR_INFO[sensor]
        root = Path(task_roots[sensor])
        if not root.exists():
            print(f'[skip] sensor {sensor}: episode_root not found {root}')
            continue
        eps = sorted(root.glob('episode_*.hdf5'))
        if not eps:
            print(f'[skip] sensor {sensor}: no episodes under {root}')
            continue
        if args.episode_indices is not None:
            chosen = [eps[i] for i in args.episode_indices if i < len(eps)]
        else:
            chosen = eps[:args.episodes_per_sensor]
        out_sensor_dir = out_root / args.task / f'sensor_{info["name"]}'
        out_sensor_dir.mkdir(parents=True, exist_ok=True)
        for ep in tqdm(chosen, desc=f'[{sensor}] viz'):
            out_path = out_sensor_dir / f'{ep.stem}_cross_decoder.mp4'
            render_episode_video(model, dataset_details, sensor, ep,
                                 out_path, args.device,
                                 fps=args.fps, max_frames=args.max_frames)


if __name__ == '__main__':
    main()
