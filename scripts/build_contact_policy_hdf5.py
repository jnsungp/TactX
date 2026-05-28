#!/usr/bin/env python3
"""Build per-frame binary contact HDF5s for ACT policy training using a
**sensor-global baseline + absolute threshold**.

For each source policy episode HDF5:
  1. Loads observation.tac0 / observation.tac1 (raw per-frame tactile).
  2. For daimon, restricts to channels [depth, shear_x, shear_y] -- the
     same channels the TACTX encoder consumes
     (scripts/preprocess_cross_modal_new.py::daimon_to_chw).
  3. Skips the episode (and logs it) if any reading is non-finite.
  4. score_t = mean_abs(x_t - baseline_sensor)
     where baseline_sensor is a per-finger fixed vector computed once by
     scripts/contact_pick_thresholds.py and saved at
     contact_thresholds/<sensor>.npz.
  5. mask_t = score_t > threshold_sensor  (single scalar per sensor, used
     uniformly across every task -- so the bit is sensor-aware but
     task-invariant, and deployable frame-by-frame at inference).
  6. Morphological close (fill short 0-gaps) then open (drop short 1-runs)
     to denoise.
  7. Writes a new HDF5 with observation.tac0/tac1 -> (T, 1) float32 binary
     mask. Every other key (action, state, cam0, cam1, image.tac0/tac1) is
     stored as an HDF5 ExternalLink back to the source file -- contact
     HDF5s are KB-sized, not GB-sized.

Default thresholds (picked from per-sensor score histograms across all 4
tasks; see contact_thresholds/<sensor>_score_hist.png):
    daimon = 0.04
    eflesh = 300
    flexi  = 0.02

Usage:
  # auto-detect sensor from source dir name (dai_dai_*, eflesh_eflesh_*,
  # flexi_flexi_*)
  python scripts/build_contact_policy_hdf5.py \\
      --src_root tactile_benchmark/data/recordings/processed/eflesh_eflesh_board_wipe_new \\
      --dst_root tactile_benchmark/data/recordings/processed/contact/eflesh_eflesh_board_wipe_new_contact

  # override threshold and/or sensor
  python scripts/build_contact_policy_hdf5.py --threshold 0.03 ...
"""

import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPLACE_KEYS = ('observation.tac0', 'observation.tac1')

# Daimon raw (240, 320, 5) flattened channel-major to 384000:
#   flat[0:N]=depth, flat[N:2N]=ch1, flat[2N:3N]=ch2,
#   flat[3N:4N]=shear_x, flat[4N:5N]=shear_y.
# We keep [depth, shear_x, shear_y] = channels [0, 3, 4] to match the
# TACTX encoder's input.
_DAIMON_FLAT = 384000
_DAIMON_N = 240 * 320
_DAIMON_KEEP = (0, 3, 4)

DEFAULT_THRESHOLDS = {
    'daimon': 0.04,
    'eflesh': 300.0,
    'flexi':  0.02,
}

SENSOR_FROM_PREFIX = {
    'dai_dai':       'daimon',
    'eflesh_eflesh': 'eflesh',
    'flexi_flexi':   'flexi',
}


def detect_sensor(src_root):
    name = os.path.basename(os.path.normpath(src_root))
    for prefix, sensor in SENSOR_FROM_PREFIX.items():
        if name.startswith(prefix):
            return sensor
    raise ValueError(
        f"can't infer sensor from dir name {name!r}; "
        f"pass --sensor one of {list(SENSOR_FROM_PREFIX.values())}")


def select_daimon_channels(x):
    """x: (T, 384000) -> (T, 3*N) keeping channels [depth, shear_x, shear_y]."""
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


def morph_close_open(mask, min_gap, min_run):
    """Fill interior 0-gaps shorter than min_gap, then drop 1-runs shorter
    than min_run. Border 0-runs are never filled."""
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


def plot_episode(ep_name, sa, sb, ma, mb, threshold, out_path, ymax=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, sc, mask, name in (
        (axes[0], sa, ma, 'tac0'),
        (axes[1], sb, mb, 'tac1'),
    ):
        fr = np.arange(len(sc))
        ax.plot(fr, sc, color='#1f77b4', lw=1.6, label=name)
        ax.axhline(threshold, ls='--', color='k', lw=1.0,
                   label=f'thr = {threshold:g}')
        first = True
        for s, e, v in _runs(mask.astype(bool)):
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
        ax.set_title(f'{ep_name}  {name}  '
                     f'contact={int(mask.sum())}/{len(mask)}')
    axes[-1].set_xlabel('frame')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def process_episode(src_path, dst_path, sensor, baseline_a, baseline_b,
                    threshold, min_gap, min_run,
                    plot_path=None, ymax=None):
    """Returns (n_contact_a, n_contact_b, T) on success, or None if the
    episode was skipped because it contained any non-finite reading."""
    src_abs = os.path.abspath(src_path)
    with h5py.File(src_path, 'r') as src:
        tac0 = np.asarray(src['observation.tac0'][()], dtype=np.float32)
        tac1 = np.asarray(src['observation.tac1'][()], dtype=np.float32)
        link_keys = [k for k in src.keys() if k not in REPLACE_KEYS]
        root_attrs = dict(src.attrs)

    if (not np.isfinite(tac0).all()) or (not np.isfinite(tac1).all()):
        return None

    if sensor == 'daimon':
        tac0 = select_daimon_channels(tac0)
        tac1 = select_daimon_channels(tac1)

    sa = np.abs(tac0 - baseline_a[None]).mean(axis=1)
    sb = np.abs(tac1 - baseline_b[None]).mean(axis=1)
    ma = morph_close_open(sa > threshold, min_gap, min_run)
    mb = morph_close_open(sb > threshold, min_gap, min_run)

    if os.path.exists(dst_path):
        os.remove(dst_path)
    with h5py.File(dst_path, 'w') as dst:
        for k, v in root_attrs.items():
            dst.attrs[k] = v
        for k in link_keys:
            dst[k] = h5py.ExternalLink(src_abs, '/' + k)
        dst.create_dataset('observation.tac0',
                           data=ma.astype(np.float32).reshape(-1, 1))
        dst.create_dataset('observation.tac1',
                           data=mb.astype(np.float32).reshape(-1, 1))

    if plot_path is not None:
        plot_episode(os.path.basename(src_path), sa, sb,
                     ma.astype(np.uint8), mb.astype(np.uint8),
                     threshold, plot_path, ymax=ymax)

    return int(ma.sum()), int(mb.sum()), len(ma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', required=True)
    ap.add_argument('--dst_root', required=True)
    ap.add_argument('--baseline_dir',
                    default='/home/junsung/cross_modal_tactile/contact_thresholds',
                    help='dir holding <sensor>.npz from contact_pick_thresholds.py')
    ap.add_argument('--sensor', choices=list(SENSOR_FROM_PREFIX.values()),
                    default=None,
                    help='auto-detected from src_root if omitted')
    ap.add_argument('--threshold', type=float, default=None,
                    help='per-sensor threshold; default from DEFAULT_THRESHOLDS')
    ap.add_argument('--smooth_min_gap', type=int, default=2)
    ap.add_argument('--smooth_min_run', type=int, default=2)
    ap.add_argument('--visualize', action='store_true',
                    help='Emit per-episode contact plots under <dst_root>/_plots')
    ap.add_argument('--plot_dir', default=None)
    ap.add_argument('--max_plots', type=int, default=5,
                    help='cap on plots per dir; 0 = no cap')
    args = ap.parse_args()

    if not os.path.isdir(args.src_root):
        print(f'[fatal] src_root does not exist: {args.src_root}', file=sys.stderr)
        sys.exit(1)

    sensor = args.sensor or detect_sensor(args.src_root)
    threshold = (args.threshold if args.threshold is not None
                 else DEFAULT_THRESHOLDS[sensor])

    npz_path = os.path.join(args.baseline_dir, f'{sensor}.npz')
    if not os.path.exists(npz_path):
        print(f'[fatal] baseline .npz missing: {npz_path}\n'
              f'  -> run scripts/contact_pick_thresholds.py first.',
              file=sys.stderr)
        sys.exit(1)
    npz = np.load(npz_path)
    baseline_a = npz['baseline_a']
    baseline_b = npz['baseline_b']
    # y-axis cap for plots: 99.5th percentile of training scores (if saved).
    if 'scores_a' in npz.files and 'scores_b' in npz.files:
        ymax = float(max(np.percentile(npz['scores_a'], 99.5),
                         np.percentile(npz['scores_b'], 99.5)))
    else:
        ymax = threshold * 5

    files = sorted(f for f in os.listdir(args.src_root) if f.endswith('.hdf5'))
    if not files:
        print(f'[fatal] no episode_*.hdf5 in {args.src_root}', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.dst_root, exist_ok=True)
    plot_dir = args.plot_dir or os.path.join(args.dst_root, '_plots')
    if args.visualize:
        os.makedirs(plot_dir, exist_ok=True)

    print(f'{args.src_root} -> {args.dst_root}')
    print(f'  sensor={sensor}  threshold={threshold:g}  '
          f'|baseline_a|={np.abs(baseline_a).mean():.4f}  '
          f'|baseline_b|={np.abs(baseline_b).mean():.4f}')

    tot_a = tot_b = tot_T = 0
    skipped = []
    for i, f in enumerate(tqdm(files)):
        do_plot = (args.visualize
                   and (args.max_plots == 0 or i < args.max_plots))
        plot_path = (os.path.join(plot_dir, f.replace('.hdf5', '.png'))
                     if do_plot else None)
        result = process_episode(
            os.path.join(args.src_root, f),
            os.path.join(args.dst_root, f),
            sensor, baseline_a, baseline_b, threshold,
            args.smooth_min_gap, args.smooth_min_run,
            plot_path=plot_path, ymax=ymax,
        )
        if result is None:
            skipped.append(f)
            continue
        a, b, T = result
        tot_a += a
        tot_b += b
        tot_T += T

    if skipped:
        print(f'  SKIPPED {len(skipped)} non-finite episodes:')
        for f in skipped:
            print(f'    {f}')
    if tot_T:
        print(f'  tac0 frac={tot_a / tot_T:.3f}'
              f'   tac1 frac={tot_b / tot_T:.3f}'
              f'   total frames={tot_T}')


if __name__ == '__main__':
    main()
