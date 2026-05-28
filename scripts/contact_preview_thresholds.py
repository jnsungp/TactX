#!/usr/bin/env python3
"""Visualize the proposed per-sensor absolute contact thresholds across all
4 tasks for each sensor.

For each (sensor, task) val dir, samples a few episodes, computes
    score_t = mean_abs(x_t - baseline_sensor_global)
using the (baseline_a, baseline_b) saved by contact_pick_thresholds.py,
applies the per-sensor threshold (and the same close/open morph as
build_contact_policy_hdf5.py), and writes a 2-panel plot per episode at:
    contact_thresholds/preview_plots/<sensor>/<task>/episode_XXXX.png

Threshold defaults: daimon=0.04, eflesh=300, flexi=0.02 (tunable via CLI).

Usage:
    python scripts/contact_preview_thresholds.py
    python scripts/contact_preview_thresholds.py --daimon 0.05 --eflesh 250 \\
        --per_task 3
"""

import argparse
import os
import random

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/junsung/cross_modal_tactile/tactile_benchmark/data/recordings/processed'
BASE_DIR = '/home/junsung/cross_modal_tactile/contact_thresholds'
OUT_DIR = os.path.join(BASE_DIR, 'preview_plots')

SENSOR_VAL_DIRS = {
    'daimon': ['dai_dai_board_wipe_new_val', 'dai_dai_spam_val',
               'dai_dai_object_pnp_val', 'dai_dai_plug_insertion_direct_val'],
    'eflesh': ['eflesh_eflesh_board_wipe_new_val', 'eflesh_eflesh_spam_val',
               'eflesh_eflesh_object_pnp_val',
               'eflesh_eflesh_plug_insertion_direct_val'],
    'flexi':  ['flexi_flexi_board_wipe_new_val', 'flexi_flexi_spam_val',
               'flexi_flexi_object_pnp_val',
               'flexi_flexi_plug_insertion_direct_val'],
}

_DAIMON_FLAT = 384000
_DAIMON_N = 240 * 320
_DAIMON_KEEP = (0, 3, 4)


def select_daimon(x):
    T = x.shape[0]
    return x.reshape(T, 5, _DAIMON_N)[:, _DAIMON_KEEP, :].reshape(T, -1)


def _runs(mask):
    out = []
    n = len(mask)
    if n == 0:
        return out
    cur = bool(mask[0])
    s = 0
    for i in range(1, n):
        if bool(mask[i]) != cur:
            out.append((s, i, cur))
            cur = bool(mask[i])
            s = i
    out.append((s, n, cur))
    return out


def morph_close_open(mask, min_gap=2, min_run=2):
    out = np.asarray(mask, dtype=bool).copy()
    n = len(out)
    if n == 0:
        return out
    for s, e, v in _runs(out):
        if (not v) and (e - s) < min_gap and s > 0 and e < n:
            out[s:e] = True
    for s, e, v in _runs(out):
        if v and (e - s) < min_run:
            out[s:e] = False
    return out


def load_episode(hdf5_path, sensor):
    with h5py.File(hdf5_path, 'r') as f:
        a = np.asarray(f['observation.tac0'][()], dtype=np.float32)
        b = np.asarray(f['observation.tac1'][()], dtype=np.float32)
    if (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
        return None, None, 'non-finite'
    if sensor == 'daimon':
        a = select_daimon(a)
        b = select_daimon(b)
    return a, b, None


def plot_episode(ep_name, sc_a, sc_b, mask_a, mask_b, thr, out_path,
                 ymax=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    pa = int(mask_a.sum())
    pb = int(mask_b.sum())
    for ax, sc, mask, name, n_pos in (
        (axes[0], sc_a, mask_a, 'tac0', pa),
        (axes[1], sc_b, mask_b, 'tac1', pb),
    ):
        fr = np.arange(len(sc))
        ax.plot(fr, sc, color='#1f77b4', lw=1.6, label=name)
        ax.axhline(thr, ls='--', color='k', lw=1.0,
                   label=f'thr = {thr:g}')
        m = mask.astype(bool)
        first = True
        for s, e, v in _runs(m):
            if not v:
                continue
            ax.axvspan(fr[s] - 0.5, fr[e - 1] + 0.5, color='gray', alpha=0.30,
                       label='contact' if first else None)
            first = False
        if ymax is not None:
            ax.set_ylim(0, ymax)
        ax.set_ylabel('mean_abs(x - baseline_global)')
        ax.legend(fontsize=8, ncol=2, loc='upper right')
        ax.grid(alpha=0.25)
        ax.set_title(f'{ep_name}  {name}  contact={n_pos}/{len(sc)}')
    axes[-1].set_xlabel('frame')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--daimon', type=float, default=0.04)
    ap.add_argument('--eflesh', type=float, default=300.0)
    ap.add_argument('--flexi',  type=float, default=0.02)
    ap.add_argument('--per_task', type=int, default=3,
                    help='Episodes per (sensor, task) to plot.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    thresholds = {'daimon': args.daimon,
                  'eflesh': args.eflesh,
                  'flexi':  args.flexi}
    rng = random.Random(args.seed)

    for sensor, task_dirs in SENSOR_VAL_DIRS.items():
        npz = np.load(os.path.join(BASE_DIR, f'{sensor}.npz'))
        baseline_a = npz['baseline_a']
        baseline_b = npz['baseline_b']
        thr = thresholds[sensor]
        # y-axis cap so the plots are comparable across episodes for a sensor;
        # use 3x threshold (or the actual max) to keep the threshold line
        # visible without flattening contact peaks.
        # Pick from training score distribution if available.
        if 'scores_a' in npz.files:
            ymax = float(max(np.percentile(npz['scores_a'], 99.5),
                             np.percentile(npz['scores_b'], 99.5)))
        else:
            ymax = thr * 5
        print(f'\n=== {sensor}  thr={thr:g}  ymax={ymax:g} ===')

        for d in task_dirs:
            src = os.path.join(ROOT, d)
            if not os.path.isdir(src):
                print(f'  [skip] {d} not found')
                continue
            files = sorted(f for f in os.listdir(src) if f.endswith('.hdf5'))
            if not files:
                print(f'  [skip] {d} empty')
                continue
            if args.per_task and len(files) > args.per_task:
                files = sorted(rng.sample(files, args.per_task))
            out_sub = os.path.join(OUT_DIR, sensor, d)
            os.makedirs(out_sub, exist_ok=True)
            kept = 0
            for f in files:
                p = os.path.join(src, f)
                a, b, reason = load_episode(p, sensor)
                if reason is not None:
                    print(f'  [skip-inf] {d}/{f}')
                    continue
                sa = np.abs(a - baseline_a[None]).mean(axis=1)
                sb = np.abs(b - baseline_b[None]).mean(axis=1)
                ma = morph_close_open(sa > thr)
                mb = morph_close_open(sb > thr)
                out_png = os.path.join(out_sub, f.replace('.hdf5', '.png'))
                plot_episode(f, sa, sb, ma, mb, thr, out_png, ymax=ymax)
                kept += 1
            print(f'  {d}: {kept} plots')


if __name__ == '__main__':
    main()
