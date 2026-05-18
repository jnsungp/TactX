"""
Render side-by-side GIFs of teleop episodes from latent_space_teleop.

Layout:
    latent_space_teleop/pnp/<combo>/tactile/episode_XXXX/<timestamp>.pkl

Each .pkl is a dict: {ts, s0, s1, [img0, img1]}.
    s0 = LEFT sensor reading, s1 = RIGHT sensor reading.
The combo name (e.g. left_daimon_right_eflesh) names which sensor is which.

Per-sensor reading shapes / renderers:
    daimon : (240, 320, 5) — multi-channel tactile image; first 3 channels
             shown as an RGB image (per-frame min-max normalized).
    eflesh : (5, 3)        — 5 magnets x XYZ, grouped bar chart.
    flexi  : (12, 16)      — heatmap (viridis).

Both active sensors (s0, s1) are drawn side by side, frame by frame.

By default only CONTACT frames are rendered — no-contact frames (sensor
activity below --contact_threshold) are dropped. Use --all_frames to keep
everything.

Usage:
    python scripts/visualize_teleop_episodes.py
    python scripts/visualize_teleop_episodes.py --combos left_daimon_right_eflesh
    python scripts/visualize_teleop_episodes.py --n_episodes 5 --fps 15
    python scripts/visualize_teleop_episodes.py --all_frames        # keep no-contact too
"""

import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


_DEFAULT_ROOT = 'tactile_benchmark/data/latent_space_teleop/pnp'
_DEFAULT_COMBOS = ['left_daimon_right_eflesh',
                   'left_eflesh_right_flexi',
                   'left_daimon_right_flexi']


def _parse_combo(combo):
    """left_daimon_right_eflesh -> ('daimon', 'eflesh')."""
    parts = combo.split('_')
    # left <A> right <B>  (sensor names are single tokens)
    try:
        li = parts.index('left')
        ri = parts.index('right')
        return parts[li + 1], parts[ri + 1]
    except (ValueError, IndexError):
        return 's0', 's1'


def _load_frame(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def _episode_frames(ep_dir):
    """Sorted frame-.pkl paths. Frame files are named <timestamp>.pkl; this
    keeps only those (numeric stem) so non-frame files like metadata.pkl —
    which has no s0/s1 — are excluded. Lexical sort = time order."""
    fs = []
    for f in glob.glob(os.path.join(ep_dir, '*.pkl')):
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            float(stem)
        except ValueError:
            continue
        fs.append(f)
    return sorted(fs)


# ---------------------------------------------------------------------------
# Per-sensor renderers
# ---------------------------------------------------------------------------

def _render_daimon(ax, arr, vlim=None):
    """arr: (H, W, C>=3). Show the first 3 channels as an RGB image
    (per-frame min-max normalized over those 3 channels)."""
    img = np.asarray(arr, dtype=np.float32)[:, :, :3]
    lo, hi = float(img.min()), float(img.max())
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.set_title('Daimon (240,320,5) — channels 0-2 (RGB)')
    ax.axis('off')


def _render_eflesh(ax, arr, vlim):
    """arr: (5, 3) — 5 magnets x XYZ."""
    v = np.asarray(arr, dtype=np.float32).reshape(5, 3)
    bar_w = 0.25
    idx = np.arange(5)
    for k, (c, lbl) in enumerate(zip(('#e74c3c', '#2ecc71', '#3498db'),
                                     ('X', 'Y', 'Z'))):
        ax.bar(idx + (k - 1) * bar_w, v[:, k], width=bar_w, color=c, label=lbl)
    ax.set_xticks(idx)
    ax.set_xticklabels([f'M{i}' for i in range(5)])
    ax.set_ylim(vlim)
    ax.set_title('eFlesh (5,3) = 5 magnets x XYZ')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)


def _render_flexi(ax, arr, vlim):
    """arr: (12, 16) heatmap."""
    m = np.asarray(arr, dtype=np.float32)
    ax.imshow(m, cmap='viridis', vmin=vlim[0], vmax=vlim[1], aspect='auto')
    ax.set_title(f'FlexiTac (12,16)  v=[{vlim[0]:.2f},{vlim[1]:.2f}]')
    ax.axis('off')


def _contact_keep(frames, threshold):
    """Keep only frames with contact. Per frame, each sensor's activity is the
    std of its reading; each sensor's activity series is normalized to [0,1]
    within the episode, and a frame is 'contact' if either sensor exceeds
    `threshold`. FlexiTac is a clean signal (~0 with no contact); Daimon /
    eFlesh activity rises with contact too, so the relative normalization
    self-calibrates per sensor.

    Returns (kept_frame_paths, n_kept, n_total).
    """
    a0, a1 = [], []
    for p in frames:
        d = _load_frame(p)
        a0.append(float(np.asarray(d['s0'], dtype=np.float32).std()))
        a1.append(float(np.asarray(d['s1'], dtype=np.float32).std()))

    def _norm(a):
        a = np.asarray(a, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)

    keep = (_norm(a0) > threshold) | (_norm(a1) > threshold)
    kept = [f for f, k in zip(frames, keep) if k]
    return kept, int(keep.sum()), len(frames)


def _episode_vlims(frames, sensor0, sensor1):
    """Sweep the episode once for stable per-sensor display ranges.
    Daimon is per-frame normalized in its renderer, so it gets no vlim."""
    def _acc(key, sensor):
        if sensor == 'daimon':
            return None
        lo, hi = float('inf'), float('-inf')
        for p in frames[::max(1, len(frames) // 30)]:
            arr = np.asarray(_load_frame(p)[key], dtype=np.float32)
            lo = min(lo, float(arr.min()))
            hi = max(hi, float(arr.max()))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        if sensor == 'eflesh':                    # symmetric for bars
            m = max(abs(lo), abs(hi)) * 1.05
            return (-m, m)
        return (lo, hi)
    return _acc('s0', sensor0), _acc('s1', sensor1)


def _fig_to_pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(rgba[..., :3], mode='RGB')


def _render(ax, sensor, arr, vlim):
    if sensor == 'daimon':
        _render_daimon(ax, arr)
    elif sensor == 'eflesh':
        _render_eflesh(ax, arr, vlim)
    elif sensor == 'flexi':
        _render_flexi(ax, arr, vlim)
    else:                                          # unknown — best effort
        a = np.asarray(arr, dtype=np.float32)
        (ax.imshow(a, aspect='auto') if a.ndim == 2
         else ax.bar(range(a.size), a.ravel()))
        ax.set_title(f'{sensor} {tuple(a.shape)}')


def render_episode_gif(combo, ep_dir, out_path, fps, max_frames,
                       contact_threshold, all_frames):
    sensor0, sensor1 = _parse_combo(combo)
    frames = _episode_frames(ep_dir)
    if not frames:
        print(f'  [skip] no frames in {ep_dir}')
        return

    if not all_frames:
        frames, n_keep, n_tot = _contact_keep(frames, contact_threshold)
        print(f'  contact frames: {n_keep}/{n_tot}')
        if not frames:
            print('  [skip] no contact frames detected')
            return

    vlim0, vlim1 = _episode_vlims(frames, sensor0, sensor1)

    if max_frames and len(frames) > max_frames:
        sel = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in sel]

    pil_frames = []
    for i, p in enumerate(frames):
        d = _load_frame(p)
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=90)
        _render(axs[0], sensor0, d['s0'], vlim0)
        _render(axs[1], sensor1, d['s1'], vlim1)
        axs[0].set_ylabel(f'LEFT  ({sensor0})', fontsize=9)
        axs[1].set_ylabel(f'RIGHT ({sensor1})', fontsize=9)
        fig.suptitle(f'{combo} | {os.path.basename(ep_dir)} | frame {i:04d}',
                     fontsize=11)
        fig.tight_layout()
        pil_frames.append(_fig_to_pil(fig))
        plt.close(fig)

    duration_ms = max(int(1000 / max(fps, 1)), 20)
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=duration_ms, loop=0, disposal=2)
    print(f'  wrote {out_path}  ({len(pil_frames)} frames @ {fps} fps)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=_DEFAULT_ROOT)
    parser.add_argument('--combos', nargs='+', default=_DEFAULT_COMBOS)
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Episodes per combo to render (from the start).')
    parser.add_argument('--out_dir', default='visualizations/teleop_episodes')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Cap on frames per GIF (uniform subsample).')
    parser.add_argument('--contact_threshold', type=float, default=0.12,
                        help='Contact-detection threshold on per-sensor '
                             'normalized activity [0,1]. Higher = stricter '
                             '(fewer frames kept).')
    parser.add_argument('--all_frames', action='store_true',
                        help='Disable contact filtering — render every frame, '
                             'including no-contact.')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = (args.data_root if os.path.isabs(args.data_root)
                 else os.path.join(repo_root, args.data_root))
    os.makedirs(args.out_dir, exist_ok=True)

    for combo in args.combos:
        tac_dir = os.path.join(data_root, combo, 'tactile')
        if not os.path.isdir(tac_dir):
            print(f'[skip] no dir: {tac_dir}')
            continue
        eps = sorted(d for d in glob.glob(os.path.join(tac_dir, 'episode_*'))
                     if os.path.isdir(d))
        eps = eps[:args.n_episodes]
        print(f'[{combo}] rendering {len(eps)} episodes')
        for ep_dir in eps:
            out = os.path.join(
                args.out_dir, f'{combo}__{os.path.basename(ep_dir)}.gif')
            render_episode_gif(combo, ep_dir, out, args.fps, args.max_frames,
                               args.contact_threshold, args.all_frames)


if __name__ == '__main__':
    main()
