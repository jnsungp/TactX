"""
Render side-by-side GIFs of random paired episodes from
latent_data_processed_object/{de_pair,ef_pair,fd_pair}/{sensor}/{object}/.

For each pair, picks N random episodes (intersection of episodes present
in both sensors' folders for the requested object) and writes one GIF per
episode: sensor_a on the left, sensor_b on the right, frame-by-frame.

Sensor renderers:
    Daimon  (3, H, W) — per-frame min-max → RGB image
    eFlesh  (15,)     — 5 magnets × 3 axes as a grouped bar chart
                        (y-limits computed once per episode)
    Flexi   (12, 16)  — heatmap (viridis), colorbar fixed to episode range

Dependencies: numpy, torch, matplotlib, Pillow. No imageio required.

Usage (defaults: 2 eps × 3 pairs × cylinder = 6 GIFs):
    python scripts/visualize_paired_episodes_v2.py
    python scripts/visualize_paired_episodes_v2.py --object cylinder --n_episodes 2 --fps 10
    python scripts/visualize_paired_episodes_v2.py --pairs de fd --object ball --seed 42
"""

import argparse
import glob
import os
import random
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


_EP_FR_RE = re.compile(r'ep(\d+)_fr(\d+)\.pt$')

_PAIR_CFG = {
    'de': ('daimon', 'eflesh'),
    'ef': ('eflesh', 'flexi'),
    'fd': ('flexi', 'daimon'),
}


def _parse_ep_fr(name):
    m = _EP_FR_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def _list_paired_episodes(pair_root, sensor_a, sensor_b, obj):
    """Return {ep_id: [(frame, path_a, path_b), ...]} for files present in BOTH
    sensors' object dir. Mirrors CrossModalPairDataset's basename-intersection
    pairing rule."""
    a_dir = os.path.join(pair_root, sensor_a, obj)
    b_dir = os.path.join(pair_root, sensor_b, obj)
    a = {os.path.basename(p): p for p in glob.glob(os.path.join(a_dir, '*.pt'))}
    b = {os.path.basename(p): p for p in glob.glob(os.path.join(b_dir, '*.pt'))}
    common = sorted(set(a) & set(b))
    episodes = defaultdict(list)
    for n in common:
        ep, fr = _parse_ep_fr(n)
        if ep < 0:
            continue
        episodes[ep].append((fr, a[n], b[n]))
    for ep in episodes:
        episodes[ep].sort(key=lambda t: t[0])
    return episodes


def _load_tensor(path):
    x = torch.load(path, weights_only=False)
    if isinstance(x, dict):
        for k in ('image', 'magnetic', 'flexi', 'flexitac'):
            if k in x:
                return x[k].float()
        return next(iter(x.values())).float()
    return x.float()


def _apply_runtime_transform(x, mode):
    """Apply 'fliplr' / 'flipud' / 'rot180' to a tensor. Works for (C,H,W),
    (H,W), or (D,) — uses negative dim indices.

    fliplr → mirror about the vertical axis (left/right swap)
    flipud → mirror about the horizontal axis (top/bottom swap)
    rot180 → both, equivalent to undoing a fliplr and applying a flipud
    """
    if mode in (None, 'none'):
        return x
    if x.dim() < 2 and mode in ('flipud', 'rot180'):
        # 1-D tensor has no vertical axis; only fliplr makes sense.
        return torch.flip(x, dims=[-1]) if mode == 'fliplr' else x
    if mode == 'fliplr':
        return torch.flip(x, dims=[-1])
    if mode == 'flipud':
        return torch.flip(x, dims=[-2])
    if mode == 'rot180':
        return torch.flip(x, dims=[-2, -1])
    raise ValueError(f"Unknown transform mode: {mode}")


def _render_daimon(ax, x):
    """x: (3, H, W). Per-frame min-max → RGB."""
    img = x.detach().cpu().numpy().transpose(1, 2, 0)
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.set_title('Daimon (3, 240, 320)')
    ax.axis('off')


def _render_eflesh(ax, x, vlim):
    """x: (15,) → 5×3 grouped bars. vlim is fixed per episode."""
    v = x.detach().cpu().numpy().reshape(5, 3)
    bar_w = 0.25
    idx = np.arange(5)
    colors = ('#e74c3c', '#2ecc71', '#3498db')
    for k, (c, lbl) in enumerate(zip(colors, ('X', 'Y', 'Z'))):
        ax.bar(idx + (k - 1) * bar_w, v[:, k], width=bar_w, color=c, label=lbl)
    ax.set_xticks(idx)
    ax.set_xticklabels([f'M{i}' for i in range(5)])
    ax.set_ylim(vlim)
    ax.set_title('eFlesh (15,) = 5 magnets × XYZ')
    ax.set_ylabel('magnetic')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)


def _render_flexi(ax, x, vlim):
    """x: (12, 16) → heatmap. vlim fixed per episode."""
    m = x.detach().cpu().numpy()
    ax.imshow(m, cmap='viridis', vmin=vlim[0], vmax=vlim[1], aspect='auto')
    ax.set_title(f'FlexiTac (12, 16)  v=[{vlim[0]:.2f}, {vlim[1]:.2f}]')
    ax.axis('off')


def _episode_vlim(paths):
    """Sweep all frames of an episode to fix a stable (vmin, vmax)."""
    lo, hi = float('inf'), float('-inf')
    for p in paths:
        v = _load_tensor(p)
        lo = min(lo, float(v.min()))
        hi = max(hi, float(v.max()))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return lo, hi


def _fig_to_pil(fig):
    """matplotlib Figure → PIL.Image (RGB). Uses buffer_rgba (matplotlib ≥3.8)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(rgba[..., :3], mode='RGB')


def _render_episode_gif(pair_key, sensor_a, sensor_b, frames, out_path,
                        fps, max_frames, stride, transform_by_sensor):
    """frames: list of (fr, path_a, path_b), already sorted by fr.

    transform_by_sensor: dict mapping sensor_name → mode string for runtime
    flip/rot to apply before rendering (and before vlim sweep).
    """
    if stride > 1:
        frames = frames[::stride]
    if max_frames is not None and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    def _load_and_transform(path, sensor):
        return _apply_runtime_transform(
            _load_tensor(path), transform_by_sensor.get(sensor, 'none')
        )

    # Stable per-episode limits for non-image sensors — sweep with transform applied.
    vlim = {}
    for side, sensor in (('a', sensor_a), ('b', sensor_b)):
        if sensor == 'daimon':
            continue
        paths = [f[1] if side == 'a' else f[2] for f in frames]
        lo, hi = float('inf'), float('-inf')
        for p in paths:
            v = _load_and_transform(p, sensor)
            lo = min(lo, float(v.min()))
            hi = max(hi, float(v.max()))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        if sensor == 'eflesh':
            m = max(abs(lo), abs(hi)) * 1.05
            vlim[side] = (-m, m)
        else:
            vlim[side] = (lo, hi)

    pil_frames = []
    for fr, pa, pb in frames:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=90)
        for ax, sensor, path, side in (
            (axs[0], sensor_a, pa, 'a'),
            (axs[1], sensor_b, pb, 'b'),
        ):
            x = _load_and_transform(path, sensor)
            if sensor == 'daimon':
                _render_daimon(ax, x)
            elif sensor == 'eflesh':
                _render_eflesh(ax, x, vlim[side])
            elif sensor == 'flexi':
                _render_flexi(ax, x, vlim[side])
        tx = transform_by_sensor
        tag = ', '.join(f"{s}={m}" for s, m in tx.items() if m != 'none')
        suptitle = f'{pair_key}_pair | fr={fr:04d}'
        if tag:
            suptitle += f'  [transform: {tag}]'
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout()
        pil_frames.append(_fig_to_pil(fig))
        plt.close(fig)

    duration_ms = max(int(1000 / max(fps, 1)), 20)
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, disposal=2,
    )
    print(f'  wrote {out_path}  ({len(pil_frames)} frames @ {fps} fps)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='latent_data_processed_object')
    parser.add_argument('--object', default='cylinder')
    parser.add_argument('--n_episodes', type=int, default=2)
    parser.add_argument('--pairs', nargs='+', default=['de', 'ef', 'fd'],
                        choices=['de', 'ef', 'fd'])
    parser.add_argument('--out_dir', default='visualizations/paired_episodes')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1,
                        help='Subsample factor applied before max_frames cap.')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Cap on frames per GIF (uniform subsample).')
    parser.add_argument('--daimon_transform',
                        choices=['none', 'fliplr', 'flipud', 'rot180'],
                        default='none',
                        help='Runtime flip/rot applied to Daimon frames before rendering. '
                             'rot180 = undo-fliplr + apply-flipud.')
    parser.add_argument('--flexi_transform',
                        choices=['none', 'fliplr', 'flipud', 'rot180'],
                        default='none',
                        help='Runtime flip/rot applied to Flexi frames before rendering.')
    parser.add_argument('--eflesh_transform',
                        choices=['none', 'fliplr'], default='none',
                        help='Runtime transform for eFlesh. Only fliplr makes sense '
                             '(reverses the 15-d vector).')
    args = parser.parse_args()
    transform_by_sensor = {
        'daimon': args.daimon_transform,
        'flexi': args.flexi_transform,
        'eflesh': args.eflesh_transform,
    }

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for pair_key in args.pairs:
        sensor_a, sensor_b = _PAIR_CFG[pair_key]
        pair_root = os.path.join(args.data_root, f'{pair_key}_pair')
        if not os.path.isdir(pair_root):
            print(f'[skip] no dir: {pair_root}')
            continue

        episodes = _list_paired_episodes(pair_root, sensor_a, sensor_b, args.object)
        if not episodes:
            print(f'[skip] no paired frames for {pair_key}/{args.object}')
            continue

        ep_ids = sorted(episodes.keys())
        chosen = random.sample(ep_ids, min(args.n_episodes, len(ep_ids)))
        print(f'[{pair_key}] {sensor_a}↔{sensor_b}: {len(ep_ids)} episodes; '
              f'rendering eps {chosen}')

        for ep in chosen:
            frames = episodes[ep]
            if not frames:
                continue
            out = os.path.join(
                args.out_dir, f'{pair_key}_{args.object}_ep{ep:04d}.gif'
            )
            _render_episode_gif(
                pair_key, sensor_a, sensor_b, frames, out,
                args.fps, args.max_frames, args.stride, transform_by_sensor,
            )


if __name__ == '__main__':
    main()
