"""
Visualize teleop sensor readings AFTER the raw->processed transform table
+ rot180 (steps 1-2 of plot_teleop_latents.py) — a sanity check that what the
t-SNE pipeline encodes looks right.

It calls the EXACT apply_transform() from plot_teleop_latents.py, so the
panels here are precisely what gets encoded, minus only the final
resize(224) + z-score (those don't change orientation/content).

Transformed shapes rendered:
    Daimon  -> (3,240,320) = [depth, shear_x, shear_y]  -> RGB image
    eFlesh  -> (15,)        -> reshaped (5,3), grouped bars
    FlexiTac-> (12,16)      -> viridis heatmap

Both sensors (s0=left, s1=right) drawn side by side, frame by frame.

Output: visualizations/teleop_transformed/<combo>__episode_XXXX.gif

Usage:
    python scripts/visualize_teleop_transformed.py
    python scripts/visualize_teleop_transformed.py --n_episodes 5
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_teleop_latents import (
    apply_transform, contact_frames, _COMBO_SPEC, _frame_files, _episode_dirs,
    _TELEOP_ROOT,
)


# ---------------------------------------------------------------------------
# Renderers for the TRANSFORMED data
# ---------------------------------------------------------------------------

def _render_daimon(ax, arr):
    """arr: (3,240,320) = [depth, shear_x, shear_y] -> RGB (per-frame min-max)."""
    img = np.asarray(arr, dtype=np.float32).transpose(1, 2, 0)   # (H,W,3)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.set_title('Daimon transformed (3,240,320)\nR=depth G=shear_x B=shear_y',
                 fontsize=9)
    ax.axis('off')


def _render_eflesh(ax, arr, vlim):
    """arr: (15,) -> reshape (5,3), grouped bars."""
    v = np.asarray(arr, dtype=np.float32).reshape(5, 3)
    idx = np.arange(5)
    for k, (c, lbl) in enumerate(zip(('#e74c3c', '#2ecc71', '#3498db'),
                                     ('X', 'Y', 'Z'))):
        ax.bar(idx + (k - 1) * 0.25, v[:, k], width=0.25, color=c, label=lbl)
    ax.set_xticks(idx)
    ax.set_xticklabels([f'M{i}' for i in range(5)])
    ax.set_ylim(vlim)
    ax.set_title('eFlesh transformed (15,) = 5 magnets x XYZ', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)


def _render_flexi(ax, arr, vlim):
    """arr: (12,16) heatmap."""
    ax.imshow(np.asarray(arr, dtype=np.float32), cmap='viridis',
              vmin=vlim[0], vmax=vlim[1], aspect='auto')
    ax.set_title(f'FlexiTac transformed (12,16)  v=[{vlim[0]:.2f},{vlim[1]:.2f}]',
                 fontsize=9)
    ax.axis('off')


def _render(ax, sensor, arr, vlim):
    if sensor == 'D':
        _render_daimon(ax, arr)
    elif sensor == 'E':
        _render_eflesh(ax, arr, vlim)
    else:
        _render_flexi(ax, arr, vlim)


def _fig_to_pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(rgba[..., :3], mode='RGB')


def render_episode_gif(combo, ep_dir, out_path, fps, max_frames,
                       contact_threshold, all_frames):
    spec = _COMBO_SPEC[combo]
    sensor0, sensor1 = spec['s0']['sensor'], spec['s1']['sensor']
    frames = _frame_files(ep_dir)
    if not frames:
        print(f'  [skip] no frames in {ep_dir}')
        return

    if not all_frames:                       # crop to contact frames
        n0 = len(frames)
        frames = contact_frames(frames, contact_threshold)
        print(f'  contact crop: {len(frames)}/{n0}')
        if not frames:
            print('  [skip] no contact frames detected')
            return

    if max_frames and len(frames) > max_frames:
        sel = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in sel]

    # Apply the transform to every (subsampled) frame up front.
    t0, t1 = [], []
    for fp in frames:
        with open(fp, 'rb') as fh:
            d = pickle.load(fh)
        t0.append(apply_transform(d['s0'], spec['s0']))
        t1.append(apply_transform(d['s1'], spec['s1']))

    # Stable per-slot display ranges (non-Daimon).
    def _vlim(arrs, sensor):
        if sensor == 'D':
            return None
        lo = min(float(a.min()) for a in arrs)
        hi = max(float(a.max()) for a in arrs)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        if sensor == 'E':
            m = max(abs(lo), abs(hi)) * 1.05
            return (-m, m)
        return (lo, hi)
    vlim0, vlim1 = _vlim(t0, sensor0), _vlim(t1, sensor1)

    pil_frames = []
    for i, (a0, a1) in enumerate(zip(t0, t1)):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), dpi=90)
        _render(axs[0], sensor0, a0, vlim0)
        _render(axs[1], sensor1, a1, vlim1)
        axs[0].set_ylabel(f'LEFT  s0 ({sensor0})', fontsize=9)
        axs[1].set_ylabel(f'RIGHT s1 ({sensor1})', fontsize=9)
        fig.suptitle(f'{combo} | {os.path.basename(ep_dir)} | frame {i:04d} '
                     f'| post transform-table + rot180', fontsize=10)
        fig.tight_layout()
        pil_frames.append(_fig_to_pil(fig))
        plt.close(fig)

    duration_ms = max(int(1000 / max(fps, 1)), 20)
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=duration_ms, loop=0, disposal=2)
    print(f'  wrote {out_path}  ({len(pil_frames)} frames @ {fps} fps)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teleop_root', default=_TELEOP_ROOT)
    parser.add_argument('--combos', nargs='+', default=list(_COMBO_SPEC.keys()))
    parser.add_argument('--n_episodes', type=int, default=3)
    parser.add_argument('--out_dir', default='visualizations/teleop_transformed')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--contact_threshold', type=float, default=0.12,
                        help='Contact-crop threshold on per-sensor normalized '
                             'activity [0,1].')
    parser.add_argument('--all_frames', action='store_true',
                        help='Disable contact cropping — render every frame.')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    teleop_root = (args.teleop_root if os.path.isabs(args.teleop_root)
                   else os.path.join(repo_root, args.teleop_root))
    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(repo_root, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    for combo in args.combos:
        combo_dir = os.path.join(teleop_root, combo)
        if not os.path.isdir(combo_dir):
            print(f'[skip] no dir: {combo_dir}')
            continue
        eps = _episode_dirs(combo_dir)[:args.n_episodes]
        print(f'[{combo}] rendering {len(eps)} episodes')
        for ep_dir in eps:
            out = os.path.join(out_dir,
                               f'{combo}__{os.path.basename(ep_dir)}.gif')
            render_episode_gif(combo, ep_dir, out, args.fps, args.max_frames,
                               args.contact_threshold, args.all_frames)

    print(f'\nGIFs in {out_dir}/')


if __name__ == '__main__':
    main()
