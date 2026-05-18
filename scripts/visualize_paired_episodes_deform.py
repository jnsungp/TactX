"""
Render side-by-side GIFs of paired episodes, with BOTH views of Daimon:
the full original 3-channel image AND the deform-only greyscale (channel 0
replicated — what the encoder sees under dataset_sachin_static_deform).

Same format/idea as visualize_paired_episodes_v2.py, but every Daimon sensor
contributes two panels:

    Daimon (3ch RGB)   |   Daimon deform-only (grey, ch0)

So a de_pair frame renders 3 panels [Daimon-RGB, Daimon-grey, eFlesh], an
fd_pair frame renders [Flexi, Daimon-RGB, Daimon-grey], and ef_pair (no
Daimon) stays at 2 panels [eFlesh, Flexi].

Loops over all objects (default: the 10 in latent_data_processed_static),
N random episodes each.

Usage (defaults: 2 eps × 3 pairs × 10 objects):
    python scripts/visualize_paired_episodes_deform.py
    python scripts/visualize_paired_episodes_deform.py --objects cylinder plug --n_episodes 2
    python scripts/visualize_paired_episodes_deform.py --daimon_transform rot180
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

_DEFAULT_OBJECTS = ['circle', 'cylinder', 'ellipse', 'eraser', 'large_ball',
                    'line', 'plug', 'small_ball', 'spam', 'triangle']


def _parse_ep_fr(name):
    m = _EP_FR_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def _list_paired_episodes(pair_root, sensor_a, sensor_b, obj):
    """{ep_id: [(frame, path_a, path_b), ...]} for files present in BOTH sensors."""
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
    """'fliplr' / 'flipud' / 'rot180' on (C,H,W), (H,W), or (D,)."""
    if mode in (None, 'none'):
        return x
    if x.dim() < 2 and mode in ('flipud', 'rot180'):
        return torch.flip(x, dims=[-1]) if mode == 'fliplr' else x
    if mode == 'fliplr':
        return torch.flip(x, dims=[-1])
    if mode == 'flipud':
        return torch.flip(x, dims=[-2])
    if mode == 'rot180':
        return torch.flip(x, dims=[-2, -1])
    raise ValueError(f"Unknown transform mode: {mode}")


def _render_daimon_rgb(ax, x, _vlim=None):
    """x: (3, H, W). Per-frame min-max over all 3 channels → RGB."""
    img = x.detach().cpu().numpy().transpose(1, 2, 0)
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.set_title('Daimon 3ch (deform+shear)')
    ax.axis('off')


def _render_daimon_grey(ax, x, _vlim=None):
    """x: (3, H, W). Channel 0 only (deform) → greyscale. This is what the
    encoder sees under dataset_sachin_static_deform (ch0 replicated 3×)."""
    ch0 = x[0].detach().cpu().numpy()
    lo, hi = ch0.min(), ch0.max()
    if hi - lo > 1e-6:
        ch0 = (ch0 - lo) / (hi - lo)
    ax.imshow(np.clip(ch0, 0.0, 1.0), cmap='gray')
    ax.set_title('Daimon deform-only (ch0, grey)')
    ax.axis('off')


def _render_eflesh(ax, x, vlim):
    """x: (15,) → 5×3 grouped bars. vlim fixed per episode."""
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


def _panels_for(sensor):
    """Return list of (render_fn, needs_vlim) panels a sensor contributes.
    Daimon → two panels (RGB + deform-grey); others → one."""
    if sensor == 'daimon':
        return [(_render_daimon_rgb, False), (_render_daimon_grey, False)]
    if sensor == 'eflesh':
        return [(_render_eflesh, True)]
    if sensor == 'flexi':
        return [(_render_flexi, True)]
    raise ValueError(sensor)


def _fig_to_pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(rgba[..., :3], mode='RGB')


def _render_episode_gif(pair_key, sensor_a, sensor_b, frames, out_path,
                        fps, max_frames, stride, transform_by_sensor):
    if stride > 1:
        frames = frames[::stride]
    if max_frames is not None and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    def _load_and_transform(path, sensor):
        return _apply_runtime_transform(
            _load_tensor(path), transform_by_sensor.get(sensor, 'none')
        )

    # Stable per-episode limits for non-image sensors.
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

    # Panel layout: sensor_a panels then sensor_b panels.
    layout = ([('a', sensor_a, fn, nv) for fn, nv in _panels_for(sensor_a)] +
              [('b', sensor_b, fn, nv) for fn, nv in _panels_for(sensor_b)])
    n_panels = len(layout)

    pil_frames = []
    for fr, pa, pb in frames:
        fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), dpi=90)
        if n_panels == 1:
            axs = [axs]
        for ax, (side, sensor, render_fn, needs_vlim) in zip(axs, layout):
            path = pa if side == 'a' else pb
            x = _load_and_transform(path, sensor)
            render_fn(ax, x, vlim.get(side) if needs_vlim else None)
        tag = ', '.join(f"{s}={m}" for s, m in transform_by_sensor.items()
                        if m != 'none')
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
    parser.add_argument('--data_root', default='latent_data_processed_static')
    parser.add_argument('--objects', nargs='+', default=_DEFAULT_OBJECTS)
    parser.add_argument('--n_episodes', type=int, default=2)
    parser.add_argument('--pairs', nargs='+', default=['de', 'ef', 'fd'],
                        choices=['de', 'ef', 'fd'])
    parser.add_argument('--out_dir', default='visualizations/paired_episodes_deform')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--daimon_transform',
                        choices=['none', 'fliplr', 'flipud', 'rot180'],
                        default='none',
                        help='Runtime flip/rot for Daimon before rendering '
                             '(rot180 matches training-time correction).')
    parser.add_argument('--flexi_transform',
                        choices=['none', 'fliplr', 'flipud', 'rot180'],
                        default='none')
    parser.add_argument('--eflesh_transform',
                        choices=['none', 'fliplr'], default='none')
    args = parser.parse_args()
    transform_by_sensor = {
        'daimon': args.daimon_transform,
        'flexi': args.flexi_transform,
        'eflesh': args.eflesh_transform,
    }

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for obj in args.objects:
        for pair_key in args.pairs:
            sensor_a, sensor_b = _PAIR_CFG[pair_key]
            pair_root = os.path.join(args.data_root, f'{pair_key}_pair')
            if not os.path.isdir(pair_root):
                print(f'[skip] no dir: {pair_root}')
                continue

            episodes = _list_paired_episodes(pair_root, sensor_a, sensor_b, obj)
            if not episodes:
                print(f'[skip] no paired frames for {pair_key}/{obj}')
                continue

            ep_ids = sorted(episodes.keys())
            chosen = random.sample(ep_ids, min(args.n_episodes, len(ep_ids)))
            print(f'[{pair_key}/{obj}] {sensor_a}↔{sensor_b}: '
                  f'{len(ep_ids)} episodes; rendering {chosen}')

            for ep in chosen:
                frames = episodes[ep]
                if not frames:
                    continue
                out = os.path.join(
                    args.out_dir, f'{pair_key}_{obj}_ep{ep:04d}.gif'
                )
                _render_episode_gif(
                    pair_key, sensor_a, sensor_b, frames, out,
                    args.fps, args.max_frames, args.stride, transform_by_sensor,
                )


if __name__ == '__main__':
    main()
