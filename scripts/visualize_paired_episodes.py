"""
Render side-by-side GIFs of random paired episodes from
latent_data_processed_object/{de,ef,fd}_pair/.

For each pair (de=Daimon↔eFlesh, ef=eFlesh↔FlexiTac, fd=FlexiTac↔Daimon),
picks N random episodes (default 2), and for each episode writes a GIF
showing the two sensors side-by-side, frame-by-frame.

Sensor renderers:
  Daimon   (3, 240, 320) → RGB image
  eFlesh   (15,)         → 5×3 grouped bar chart (5 magnets × XYZ axes)
  FlexiTac (12, 16)      → heatmap (viridis)

Usage:
  python scripts/visualize_paired_episodes.py \
      --object cylinder \
      --n_episodes 2 \
      --pairs de ef fd \
      --out_dir visualizations/paired_episodes \
      --fps 10
"""

import argparse
import glob
import os
import random
import re
from collections import defaultdict

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


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
    """Return {ep_id: sorted_list_of_(frame, path_a, path_b)} for frames that
    exist in BOTH sensors' object dir."""
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


def _render_daimon(ax, x):
    """x: (3, H, W) float tensor. Per-frame min-max normalize to [0,1]."""
    img = x.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    img = np.clip(img, 0.0, 1.0)
    ax.imshow(img)
    ax.set_title('Daimon (3, 240, 320)')
    ax.axis('off')


def _render_eflesh(ax, x, vlim):
    """x: (15,) → 5 magnets × 3 axes. vlim is (vmin, vmax) over the episode."""
    v = x.detach().cpu().numpy().reshape(5, 3)
    n_mag, n_axis = v.shape
    bar_w = 0.25
    idx = np.arange(n_mag)
    colors = ('#e74c3c', '#2ecc71', '#3498db')   # X, Y, Z
    labels = ('X', 'Y', 'Z')
    for k in range(n_axis):
        ax.bar(idx + (k - 1) * bar_w, v[:, k], width=bar_w,
               color=colors[k], label=labels[k])
    ax.set_xticks(idx)
    ax.set_xticklabels([f'M{i}' for i in range(n_mag)])
    ax.set_ylim(vlim)
    ax.set_title('eFlesh (15,) = 5 magnets × XYZ')
    ax.set_ylabel('magnetic')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)


def _render_flexi(ax, x, vlim):
    """x: (12, 16) heatmap. vlim is (vmin, vmax) over the episode."""
    m = x.detach().cpu().numpy()
    im = ax.imshow(m, cmap='viridis', vmin=vlim[0], vmax=vlim[1], aspect='auto')
    ax.set_title('FlexiTac (12, 16)')
    ax.axis('off')
    return im


def _sensor_vlim(paths, key):
    """Find a stable (vmin, vmax) across an episode for the given sensor."""
    lo, hi = float('inf'), float('-inf')
    for p in paths:
        x = torch.load(p, weights_only=False)
        if isinstance(x, dict):
            x = x.get(key, next(iter(x.values())))
        v = x.float()
        lo = min(lo, float(v.min()))
        hi = max(hi, float(v.max()))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return lo, hi


def _load_tensor(path):
    x = torch.load(path, weights_only=False)
    if isinstance(x, dict):
        # backward compat for dict-style frames
        for k in ('image', 'magnetic', 'flexi', 'flexitac'):
            if k in x:
                return x[k].float()
        return next(iter(x.values())).float()
    return x.float()


def _fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def _render_episode_gif(pair_key, sensor_a, sensor_b, frames,
                        out_path, fps, max_frames, stride):
    """frames: list of (fr, path_a, path_b), sorted by fr.

    Strides through frames so the GIF is reasonable size; respects max_frames.
    """
    if stride > 1:
        frames = frames[::stride]
    if max_frames is not None and len(frames) > max_frames:
        # uniformly subsample to max_frames
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    # Compute stable vmin/vmax for non-image sensors over the episode.
    vlim_per_side = {}
    for side, sensor in (('a', sensor_a), ('b', sensor_b)):
        if sensor == 'daimon':
            continue
        paths = [f[1] if side == 'a' else f[2] for f in frames]
        key = {'eflesh': 'magnetic', 'flexi': 'flexi'}.get(sensor, sensor)
        vlim_per_side[side] = _sensor_vlim(paths, key)

    rgb_frames = []
    for fr, pa, pb in frames:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, sensor, path, side in (
            (axs[0], sensor_a, pa, 'a'),
            (axs[1], sensor_b, pb, 'b'),
        ):
            x = _load_tensor(path)
            if sensor == 'daimon':
                _render_daimon(ax, x)
            elif sensor == 'eflesh':
                _render_eflesh(ax, x, vlim_per_side[side])
            elif sensor == 'flexi':
                _render_flexi(ax, x, vlim_per_side[side])
            else:
                ax.set_title(f'{sensor} (unknown)')
                ax.axis('off')
        fig.suptitle(f'{pair_key} | fr={fr:04d}', fontsize=11)
        fig.tight_layout()
        rgb_frames.append(_fig_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(out_path, rgb_frames, fps=fps)
    print(f'  wrote {out_path}  ({len(rgb_frames)} frames @ {fps} fps)')


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
                        help='Subsample factor before max_frames cap.')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Cap on frames per GIF (uniform subsample).')
    args = parser.parse_args()

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
        n = min(args.n_episodes, len(ep_ids))
        chosen = random.sample(ep_ids, n)
        print(f'[{pair_key}] {sensor_a}↔{sensor_b}: {len(ep_ids)} episodes available, '
              f'rendering eps {chosen}')

        for ep in chosen:
            frames = episodes[ep]
            if not frames:
                continue
            out = os.path.join(
                args.out_dir,
                f'{pair_key}_{args.object}_ep{ep:04d}.gif',
            )
            _render_episode_gif(pair_key, sensor_a, sensor_b, frames,
                                out, args.fps, args.max_frames, args.stride)


if __name__ == '__main__':
    main()
