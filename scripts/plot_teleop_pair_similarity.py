"""
Per-episode matched-pair latent similarity for teleop data.

In each teleop combo, s0[i] and s1[i] are the SAME contact frame seen by two
sensors. If the cross-modal latent is shared their latents coincide. This
plots, per episode, the mean (+/- std) cosine similarity between the two
sensors' latents over that episode's contact frames.

One subplot per combo; x-axis = episode, bar = mean cosine sim, error bar =
std across the episode's contact frames. A dashed line marks the combo's
overall mean.

Reuses the exact transform / contact-crop / preprocess pipeline of
plot_teleop_latents.py.

Usage:
    python scripts/plot_teleop_pair_similarity.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_static_run_scratch11_shear_efs3_20260517_043942 \
        --device cuda:0
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_teleop_latents import (
    apply_transform, contact_frames, preprocess_batch, _load_model,
    _COMBO_SPEC, _frame_files, _episode_dirs, _TELEOP_ROOT,
)


@torch.no_grad()
def _encode(arrs, sensor, model, stats, image_size, device, batch_size):
    mus = []
    for s in range(0, len(arrs), batch_size):
        x = preprocess_batch(arrs[s:s + batch_size], sensor, stats,
                             image_size).to(device)
        _, mu, _, _ = model._enc(x, sensor)
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0).astype(np.float32)


@torch.no_grad()
def episode_pair_cos(frames, combo, model, stats, image_size, device,
                     batch_size):
    """Per-frame cosine similarity between the two sensors' latents for one
    episode's (already contact-cropped) frames. s0[i] / s1[i] are paired."""
    spec = _COMBO_SPEC[combo]
    sa, sb = spec['s0']['sensor'], spec['s1']['sensor']
    arr0, arr1 = [], []
    for fp in frames:
        with open(fp, 'rb') as fh:
            d = pickle.load(fh)
        arr0.append(apply_transform(d['s0'], spec['s0']))
        arr1.append(apply_transform(d['s1'], spec['s1']))
    if not arr0:
        return np.array([], dtype=np.float32)
    mu_a = _encode(arr0, sa, model, stats, image_size, device, batch_size)
    mu_b = _encode(arr1, sb, model, stats, image_size, device, batch_size)
    an = mu_a / (np.linalg.norm(mu_a, axis=1, keepdims=True) + 1e-8)
    bn = mu_b / (np.linalg.norm(mu_b, axis=1, keepdims=True) + 1e-8)
    return (an * bn).sum(axis=1).astype(np.float32)        # cosine per frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--weights', default='model.pt')
    parser.add_argument('--teleop_root', default=_TELEOP_ROOT)
    parser.add_argument('--stats_dataset', default='dataset_sachin_static')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ef_stride', type=int, default=3,
                        help='Per-episode stride for the ef combo.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--contact_threshold', type=float, default=0.12)
    parser.add_argument('--all_frames', action='store_true',
                        help='Use all frames, not just contact frames.')
    parser.add_argument('--out', default='teleop_pair_similarity.png')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    teleop_root = (args.teleop_root if os.path.isabs(args.teleop_root)
                   else os.path.join(repo_root, args.teleop_root))

    with open(os.path.join(repo_root, 'config',
                           'datasets_config_sachin.yaml')) as f:
        dscfg = yaml.safe_load(f)[args.stats_dataset]
    stats = {'daimon': dscfg['daimon_stats'],
             'eflesh': dscfg['eflesh_stats'],
             'flexi': dscfg['flexitac_stats']}
    image_size = dscfg.get('image_size', 224)

    model = _load_model(args.checkpoint_dir, args.weights, args.device)
    print(f"Loaded {args.checkpoint_dir}/{args.weights}\n")

    combos = list(_COMBO_SPEC.keys())
    fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 4.6),
                             squeeze=False)
    axes = axes[0]

    for ax, combo in zip(axes, combos):
        spec = _COMBO_SPEC[combo]
        sa, sb = spec['s0']['sensor'], spec['s1']['sensor']
        combo_dir = os.path.join(teleop_root, combo)
        is_ef = (combo == 'left_eflesh_right_flexi')
        stride = args.ef_stride if is_ef else 1

        ep_labels, means, stds, counts, all_cos = [], [], [], [], []
        for ep in _episode_dirs(combo_dir):
            frames = _frame_files(ep)
            if not args.all_frames:
                frames = contact_frames(frames, args.contact_threshold)
            frames = frames[::stride]
            if not frames:
                continue
            cos = episode_pair_cos(frames, combo, model, stats, image_size,
                                   args.device, args.batch_size)
            if cos.size == 0:
                continue
            ep_labels.append(os.path.basename(ep).replace('episode_', 'ep'))
            means.append(float(cos.mean()))
            stds.append(float(cos.std()))
            counts.append(cos.size)
            all_cos.append(cos)
            print(f"  {combo} {ep_labels[-1]}: "
                  f"cos={means[-1]:+.4f} +/- {stds[-1]:.4f}  (n={cos.size})")

        if not means:
            ax.set_title(f"{combo}\n(no frames)")
            continue

        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, capsize=3, color='#4C72B0', alpha=0.85)
        for i, (m, c) in enumerate(zip(means, counts)):
            ax.text(i, m, f'{m:+.2f}\nn={c}', ha='center',
                    va='bottom' if m >= 0 else 'top', fontsize=7)
        overall = float(np.concatenate(all_cos).mean())
        ax.axhline(overall, color='crimson', ls='--', lw=1.2,
                   label=f'combo mean {overall:+.3f}')
        ax.set_xticks(x)
        ax.set_xticklabels(ep_labels, fontsize=8)
        ax.set_ylabel('mean cosine similarity')
        ax.set_ylim(min(-0.1, min(means) - 0.15), 1.0)
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_title(f'{combo}\n{sa} vs {sb} latent similarity', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Teleop per-episode matched-pair latent similarity '
                 f"— {os.path.basename(args.checkpoint_dir.rstrip('/'))}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
