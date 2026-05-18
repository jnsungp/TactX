"""
Visualize the contact-detection threshold on teleop episodes.

contact_frames() (in plot_teleop_latents.py) keeps a frame if EITHER sensor's
per-frame activity — std of the raw reading, normalized to [0,1] within the
episode — exceeds --contact_threshold. This script plots that activity signal
per episode with the threshold line, so you can SEE which frames are contact
(kept) vs non-contact (dropped) and tune the threshold.

Per combo: one figure, one subplot per episode.
  - blue / orange line = normalized activity of s0 / s1
  - red dashed line    = threshold
  - shaded band        = NON-CONTACT region (both sensors below threshold)

Output: visualizations/contact_threshold/<combo>_threshold.png

Usage:
    python scripts/plot_contact_threshold.py
    python scripts/plot_contact_threshold.py --contact_threshold 0.2
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_teleop_latents import (
    _COMBO_SPEC, _frame_files, _episode_dirs, _TELEOP_ROOT,
)


def _episode_activity(frames):
    """Per-frame normalized activity for s0 and s1 over one episode.
    Activity = std of the raw reading; normalized to [0,1] within the episode
    (exactly what contact_frames uses)."""
    a0, a1 = [], []
    for p in frames:
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
        a0.append(float(np.asarray(d['s0'], dtype=np.float32).std()))
        a1.append(float(np.asarray(d['s1'], dtype=np.float32).std()))

    def _norm(a):
        a = np.asarray(a, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)

    return _norm(a0), _norm(a1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teleop_root', default=_TELEOP_ROOT)
    parser.add_argument('--combos', nargs='+', default=list(_COMBO_SPEC.keys()))
    parser.add_argument('--contact_threshold', type=float, default=0.12)
    parser.add_argument('--out_dir', default='visualizations/contact_threshold')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    teleop_root = (args.teleop_root if os.path.isabs(args.teleop_root)
                   else os.path.join(repo_root, args.teleop_root))
    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(repo_root, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    thr = args.contact_threshold

    for combo in args.combos:
        spec = _COMBO_SPEC[combo]
        sa, sb = spec['s0']['sensor'], spec['s1']['sensor']
        combo_dir = os.path.join(teleop_root, combo)
        if not os.path.isdir(combo_dir):
            print(f'[skip] no dir: {combo_dir}')
            continue
        eps = _episode_dirs(combo_dir)
        if not eps:
            print(f'[skip] no episodes for {combo}')
            continue

        fig, axes = plt.subplots(len(eps), 1, figsize=(11, 2.4 * len(eps)),
                                 squeeze=False)
        axes = axes[:, 0]
        for ax, ep in zip(axes, eps):
            frames = _frame_files(ep)
            if not frames:
                ax.set_title(f'{os.path.basename(ep)} (no frames)')
                continue
            n0, n1 = _episode_activity(frames)
            keep = (n0 > thr) | (n1 > thr)
            x = np.arange(len(frames))

            ax.plot(x, n0, color='#1f77b4', lw=1.0, label=f's0 ({sa})')
            ax.plot(x, n1, color='#ff7f0e', lw=1.0, label=f's1 ({sb})')
            ax.axhline(thr, color='red', ls='--', lw=1.2,
                       label=f'threshold {thr:g}')
            # shade non-contact (both below threshold)
            ax.fill_between(x, 0, 1, where=~keep, color='gray', alpha=0.25,
                            step='mid', label='non-contact')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, max(1, len(frames) - 1))
            ax.set_ylabel('norm activity', fontsize=8)
            ax.set_title(f'{os.path.basename(ep)} — contact {int(keep.sum())}'
                         f'/{len(frames)}', fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, loc='upper right', ncol=4)
        axes[-1].set_xlabel('frame')
        fig.suptitle(f'{combo} — contact threshold (shaded = non-contact, '
                     f'dropped)', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(out_dir, f'{combo}_threshold.png')
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f'  wrote {out}')

    print(f'\nFigures in {out_dir}/')


if __name__ == '__main__':
    main()
