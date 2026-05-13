"""
Render single-sensor GIFs of random episodes from
latent_data_processed_object/dynamic_data/{daimon,eflesh,flexi}/ep****_fr****.pt

For each requested sensor, picks N random episodes (default 2) and writes
one GIF per (sensor, episode). Sensor renderers mirror
visualize_paired_episodes_v2.py.

Usage:
    python scripts/visualize_dynamic_episodes.py
    python scripts/visualize_dynamic_episodes.py --n_episodes 2 --sensors daimon eflesh flexi
    python scripts/visualize_dynamic_episodes.py --seed 7 --fps 10
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


def _parse_ep_fr(name):
    m = _EP_FR_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def _list_episodes(sensor_dir):
    """{ep_id: [(frame, path), ...]} sorted by frame."""
    episodes = defaultdict(list)
    for p in glob.glob(os.path.join(sensor_dir, '*.pt')):
        ep, fr = _parse_ep_fr(os.path.basename(p))
        if ep < 0:
            continue
        episodes[ep].append((fr, p))
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


def _render_daimon(ax, x):
    img = x.detach().cpu().numpy().transpose(1, 2, 0)
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.set_title(f'Daimon  shape={tuple(x.shape)}')
    ax.axis('off')


def _render_eflesh(ax, x, vlim):
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
    m = x.detach().cpu().numpy()
    ax.imshow(m, cmap='viridis', vmin=vlim[0], vmax=vlim[1], aspect='auto')
    ax.set_title(f'FlexiTac (12, 16)  v=[{vlim[0]:.2f}, {vlim[1]:.2f}]')
    ax.axis('off')


def _episode_vlim(paths, sensor):
    lo, hi = float('inf'), float('-inf')
    for p in paths:
        v = _load_tensor(p)
        lo = min(lo, float(v.min()))
        hi = max(hi, float(v.max()))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    if sensor == 'eflesh':
        m = max(abs(lo), abs(hi)) * 1.05
        return (-m, m)
    return (lo, hi)


def _fig_to_pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(rgba[..., :3], mode='RGB')


def _render_episode_gif(sensor, ep, frames, out_path, fps, max_frames, stride):
    if stride > 1:
        frames = frames[::stride]
    if max_frames is not None and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    vlim = None
    if sensor in ('eflesh', 'flexi'):
        vlim = _episode_vlim([p for _, p in frames], sensor)

    pil_frames = []
    for fr, p in frames:
        x = _load_tensor(p)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=90)
        if sensor == 'daimon':
            _render_daimon(ax, x)
        elif sensor == 'eflesh':
            _render_eflesh(ax, x, vlim)
        elif sensor == 'flexi':
            _render_flexi(ax, x, vlim)
        fig.suptitle(f'{sensor}/dynamic_data | ep={ep:04d}  fr={fr:04d}', fontsize=11)
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
    parser.add_argument('--data_root', default='latent_data_processed_object/dynamic_data')
    parser.add_argument('--sensors', nargs='+',
                        default=['daimon', 'eflesh', 'flexi'],
                        choices=['daimon', 'eflesh', 'flexi'])
    parser.add_argument('--n_episodes', type=int, default=2)
    parser.add_argument('--out_dir', default='visualizations/dynamic_episodes')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for sensor in args.sensors:
        sensor_dir = os.path.join(args.data_root, sensor)
        if not os.path.isdir(sensor_dir):
            print(f'[skip] no dir: {sensor_dir}')
            continue

        episodes = _list_episodes(sensor_dir)
        if not episodes:
            print(f'[skip] no episodes in {sensor_dir}')
            continue

        ep_ids = sorted(episodes.keys())
        chosen = random.sample(ep_ids, min(args.n_episodes, len(ep_ids)))
        print(f'[{sensor}] {len(ep_ids)} episodes total; rendering {sorted(chosen)}')

        for ep in sorted(chosen):
            out_path = os.path.join(
                args.out_dir, f'{sensor}_ep{ep:04d}.gif'
            )
            _render_episode_gif(
                sensor, ep, episodes[ep], out_path,
                fps=args.fps, max_frames=args.max_frames, stride=args.stride,
            )


if __name__ == '__main__':
    main()
