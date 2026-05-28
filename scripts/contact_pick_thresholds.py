#!/usr/bin/env python3
"""Compute per-sensor global baseline + score distribution to pick absolute
contact thresholds.

For each sensor type (daimon, eflesh, flexi) and across BOTH board_wipe and
spam train sets:
  1. baseline_sensor = mean over (first frame of every training episode)
     -- a single fixed vector per sensor, sensor- and task-invariant.
  2. score_t = mean_abs(x_t - baseline_sensor)
     -- same channel handling as build_contact_policy_hdf5.py (daimon ->
        channels [depth, shear_x, shear_y]).
  3. Plot histogram of scores across all training frames per sensor (log y)
     so we can pick the valley between idle and contact modes by eye.
  4. Save (baseline, scores) to a .npz per sensor under
     contact_thresholds/<sensor>.npz for the labeling pass.

Usage:
    python scripts/contact_pick_thresholds.py
"""

import argparse
import os
import random
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = '/home/junsung/cross_modal_tactile/tactile_benchmark/data/recordings/processed'
OUT_DIR = '/home/junsung/cross_modal_tactile/contact_thresholds'
os.makedirs(OUT_DIR, exist_ok=True)

# All 4 tasks per sensor -- the per-sensor (baseline, threshold) we pick
# from this distribution will be applied uniformly across every task.
SENSOR_TRAIN_DIRS = {
    'daimon': ['dai_dai_board_wipe_new', 'dai_dai_spam',
               'dai_dai_object_pnp', 'dai_dai_plug_insertion_direct'],
    'eflesh': ['eflesh_eflesh_board_wipe_new', 'eflesh_eflesh_spam',
               'eflesh_eflesh_object_pnp', 'eflesh_eflesh_plug_insertion_direct'],
    'flexi':  ['flexi_flexi_board_wipe_new', 'flexi_flexi_spam',
               'flexi_flexi_object_pnp', 'flexi_flexi_plug_insertion_direct'],
}

_DAIMON_FLAT = 384000
_DAIMON_N = 240 * 320
_DAIMON_KEEP = (0, 3, 4)


def select_daimon(x):
    """x: (T, 384000) -> (T, 3*N) keeping channels [0, 3, 4]."""
    T = x.shape[0]
    return x.reshape(T, 5, _DAIMON_N)[:, _DAIMON_KEEP, :].reshape(T, -1)


def load_episode_tac(hdf5_path, sensor):
    """Return (tac0, tac1) as (T, D) float arrays, or (None, None, reason)
    if the episode contains any non-finite (NaN / inf) reading."""
    with h5py.File(hdf5_path, 'r') as f:
        a = np.asarray(f['observation.tac0'][()], dtype=np.float32)
        b = np.asarray(f['observation.tac1'][()], dtype=np.float32)
    bad_a = (~np.isfinite(a)).sum()
    bad_b = (~np.isfinite(b)).sum()
    if bad_a or bad_b:
        return None, None, f'{bad_a} non-finite in tac0, {bad_b} in tac1'
    if sensor == 'daimon':
        a = select_daimon(a)
        b = select_daimon(b)
    return a, b, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_task', type=int, default=5,
                    help='Episodes to sample per task. 0 = use all.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for sensor, dirs in SENSOR_TRAIN_DIRS.items():
        print(f'\n=== {sensor} ===')
        # Pass 1: collect first frames for global baseline.
        first_frames_a, first_frames_b = [], []
        ep_paths = []
        for d in dirs:
            src = os.path.join(ROOT, d)
            files = sorted(f for f in os.listdir(src) if f.endswith('.hdf5'))
            if args.per_task and len(files) > args.per_task:
                files = sorted(rng.sample(files, args.per_task))
            for f in files:
                ep_paths.append((d, os.path.join(src, f)))
        print(f'  {len(ep_paths)} episodes  (sampled {args.per_task} per task)')

        skipped = []
        valid_paths = []
        for d, p in tqdm(ep_paths, desc=f'{sensor} baseline'):
            a, b, reason = load_episode_tac(p, sensor)
            if reason is not None:
                skipped.append((d, p, reason))
                continue
            if a.shape[0] == 0 or b.shape[0] == 0:
                skipped.append((d, p, 'empty episode'))
                continue
            first_frames_a.append(a[0])
            first_frames_b.append(b[0])
            valid_paths.append((d, p))
        if skipped:
            print(f'  SKIPPED {len(skipped)} episodes with non-finite values:')
            for d, p, reason in skipped:
                print(f'    {os.path.relpath(p, ROOT)}  ({reason})')
        ep_paths = valid_paths
        print(f'  kept {len(ep_paths)} valid episodes')
        baseline_a = np.stack(first_frames_a, axis=0).mean(axis=0)
        baseline_b = np.stack(first_frames_b, axis=0).mean(axis=0)
        # tac0 and tac1 are same-sensor (left/right finger) but mounted
        # separately, so they may have different idle offsets. Keep separate
        # baselines per finger.
        print(f'  baseline_a shape: {baseline_a.shape}'
              f'   |baseline_a| mean: {np.abs(baseline_a).mean():.4f}')
        print(f'  baseline_b shape: {baseline_b.shape}'
              f'   |baseline_b| mean: {np.abs(baseline_b).mean():.4f}')

        # Pass 2: compute scores against the per-finger global baseline.
        # Already filtered ep_paths to finite-only episodes.
        scores_a_all, scores_b_all = [], []
        for _, p in tqdm(ep_paths, desc=f'{sensor} score'):
            a, b, _ = load_episode_tac(p, sensor)
            sa = np.abs(a - baseline_a[None]).mean(axis=1)
            sb = np.abs(b - baseline_b[None]).mean(axis=1)
            scores_a_all.append(sa.astype(np.float32))
            scores_b_all.append(sb.astype(np.float32))
        scores_a_all = np.concatenate(scores_a_all)
        scores_b_all = np.concatenate(scores_b_all)

        print(f'  scores_a: N={len(scores_a_all)}'
              f'   min={scores_a_all.min():.4f}'
              f'   max={scores_a_all.max():.4f}'
              f'   p50={np.percentile(scores_a_all, 50):.4f}'
              f'   p90={np.percentile(scores_a_all, 90):.4f}')
        print(f'  scores_b: N={len(scores_b_all)}'
              f'   min={scores_b_all.min():.4f}'
              f'   max={scores_b_all.max():.4f}'
              f'   p50={np.percentile(scores_b_all, 50):.4f}'
              f'   p90={np.percentile(scores_b_all, 90):.4f}')

        # Save baseline + raw scores for later reuse.
        np.savez(os.path.join(OUT_DIR, f'{sensor}.npz'),
                 baseline_a=baseline_a, baseline_b=baseline_b,
                 scores_a=scores_a_all, scores_b=scores_b_all)

        # Histogram plot (log y) so the idle/contact valley is visible.
        # Clip x-axis to the 99.5th percentile to drop a long flat tail of
        # high-contact frames that would otherwise compress the idle peak.
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        xmax_a = float(np.percentile(scores_a_all, 99.5))
        xmax_b = float(np.percentile(scores_b_all, 99.5))
        xmax = max(xmax_a, xmax_b)
        for ax, sc, name in ((axes[0], scores_a_all, 'tac0 (a)'),
                             (axes[1], scores_b_all, 'tac1 (b)')):
            clipped = np.clip(sc, 0.0, xmax)
            ax.hist(clipped, bins=200, log=True, color='#1f77b4', alpha=0.85,
                    range=(0.0, xmax))
            ax.axvline(np.percentile(sc, 50), color='gray', ls=':',
                       label=f"p50={np.percentile(sc, 50):.3f}")
            ax.axvline(np.percentile(sc, 75), color='orange', ls=':',
                       label=f"p75={np.percentile(sc, 75):.3f}")
            ax.axvline(np.percentile(sc, 90), color='red', ls=':',
                       label=f"p90={np.percentile(sc, 90):.3f}")
            ax.set_title(f'{sensor} {name}  '
                         f'(N={len(sc)} frames, max={sc.max():.3f}, '
                         f'shown <= p99.5={xmax:.3f})')
            ax.set_ylabel('frame count (log)')
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel('mean_abs(x - baseline_global)')
        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, f'{sensor}_score_hist.png')
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f'  -> {out_png}')

        # Per-task score percentiles so we can see if any task's contact
        # mode lives in a totally different range than the others.
        ep_offsets = {}
        idx0 = 0
        for d, p in ep_paths:
            with h5py.File(p, 'r') as f:
                n = f['observation.tac0'].shape[0]
            ep_offsets.setdefault(d, []).append((idx0, idx0 + n))
            idx0 += n
        print('  per-task score percentiles (tac0):')
        for d, ranges in ep_offsets.items():
            mask = np.zeros(len(scores_a_all), dtype=bool)
            for s, e in ranges:
                mask[s:e] = True
            sub = scores_a_all[mask]
            print(f'    {d:40s}  N={len(sub):6d}  '
                  f'p10={np.percentile(sub, 10):.3f}  '
                  f'p50={np.percentile(sub, 50):.3f}  '
                  f'p90={np.percentile(sub, 90):.3f}  '
                  f'max={sub.max():.3f}')


if __name__ == '__main__':
    main()
