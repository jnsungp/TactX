"""
Per-episode matched-pair latent cosine similarity for a sachin processed
dataset (de/ef/fd pairs, .pt format) — e.g. dataset_sachin_dynamic_pnp,
dataset_sachin_static.

In each pair, files_a[i] / files_b[i] are the SAME contact frame seen by two
sensors (basename-intersection pairing). If the cross-modal latent is shared
their latents coincide. This encodes both, computes per-frame cosine
similarity, groups by (object, episode), and plots mean (+/- std) per episode.

One subplot per pair (de / ef / fd); x = episode, bar = mean cosine sim,
error bar = std, dashed line = pair overall mean.

Usage:
    python scripts/plot_sachin_pair_similarity.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_static_run_scratch11_shear_efs3_20260517_043942 \
        --dataset_name dataset_sachin_dynamic_pnp --objects pnp \
        --ef_stride 3 --device cuda:0
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.data.datasets import CrossModalPairDataset
from joint_embedding_learning.data.datasets_sachin import (
    _build_sensor_transforms_sachin,
)
from joint_embedding_learning.utils.models_setup import get_model


def _load_model(checkpoint_dir, weights, device):
    with open(os.path.join(checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model('cross_modal_vae3', model_details).to(device)
    ckpt = torch.load(os.path.join(checkpoint_dir, weights),
                      map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    return model


def _build_pair_dataset(dd, data_root, pair_key, pair_cfg, image_size,
                        objects, ef_stride):
    """Build the CrossModalPairDataset for one pair (no train/val split)."""
    sa, sb = pair_cfg['sensor_a'], pair_cfg['sensor_b']
    pair_root = os.path.join(data_root, pair_cfg['root'])
    ta, ma, sda = _build_sensor_transforms_sachin(
        dd, sa, image_size, extra_names=pair_cfg.get('extra_transforms_a'))
    tb, mb, sdb = _build_sensor_transforms_sachin(
        dd, sb, image_size, extra_names=pair_cfg.get('extra_transforms_b'))
    stride = ef_stride if pair_key == 'ef' else 1
    ds = CrossModalPairDataset(
        pair_root, sa, sb, transforms_a=ta, transforms_b=tb,
        mean_a=ma, std_a=sda, mean_b=mb, std_b=sdb,
        device='cpu', objects=objects, frame_stride=stride,
    )
    return ds, sa, sb


@torch.no_grad()
def encode_pair(ds, pair_key, model, device, batch_size):
    """Encode all frames of a pair. Returns per-frame cosine sim + (obj, ep)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=False)
    mu_a, mu_b, objs, eps = [], [], [], []
    for (xa, xb), labels in loader:
        out = model.forward_pair(xa.to(device), xb.to(device), pair_key)
        mu_a.append(out['mu_a'].cpu().numpy())
        mu_b.append(out['mu_b'].cpu().numpy())
        objs.append(labels[0].cpu().numpy())          # object_id
        eps.append(labels[1].cpu().numpy())           # episode_id
    if not mu_a:
        return np.array([]), np.array([]), np.array([])
    a = np.concatenate(mu_a).astype(np.float32)
    b = np.concatenate(mu_b).astype(np.float32)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cos = (an * bn).sum(axis=1)
    return cos, np.concatenate(objs), np.concatenate(eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--weights', default='model.pt')
    parser.add_argument('--dataset_name', default='dataset_sachin_dynamic_pnp')
    parser.add_argument('--objects', nargs='+', default=['pnp'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ef_stride', type=int, default=3,
                        help='Per-episode stride for the ef pair (match training).')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out', default='sachin_pair_similarity.png')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, 'config',
                           'datasets_config_sachin.yaml')) as f:
        dd = yaml.safe_load(f)[args.dataset_name]
    data_root = dd['data_root']
    if not os.path.isabs(data_root):
        data_root = os.path.join(repo_root, data_root)
    image_size = dd.get('image_size', 224)
    pairs = dd['pairs']

    model = _load_model(args.checkpoint_dir, args.weights, args.device)
    print(f"Loaded {args.checkpoint_dir}/{args.weights}\n")

    pair_keys = [k for k in ('de', 'ef', 'fd') if k in pairs]
    fig, axes = plt.subplots(1, len(pair_keys),
                             figsize=(6 * len(pair_keys), 4.6), squeeze=False)
    axes = axes[0]

    for ax, pk in zip(axes, pair_keys):
        ds, sa, sb = _build_pair_dataset(dd, data_root, pk, pairs[pk],
                                         image_size, args.objects,
                                         args.ef_stride)
        cos, objs, eps = encode_pair(ds, pk, model, args.device,
                                     args.batch_size)
        if cos.size == 0:
            ax.set_title(f'{pk} ({sa} vs {sb})\n(no data)')
            continue

        # group by (object, episode)
        by_ep = defaultdict(list)
        for c, o, e in zip(cos, objs, eps):
            by_ep[(int(o), int(e))].append(float(c))
        keys = sorted(by_ep)
        labels = [f'o{o}e{e}' if len(args.objects) > 1 else f'ep{e}'
                  for (o, e) in keys]
        means = [float(np.mean(by_ep[k])) for k in keys]
        stds = [float(np.std(by_ep[k])) for k in keys]
        counts = [len(by_ep[k]) for k in keys]

        x = np.arange(len(keys))
        ax.bar(x, means, yerr=stds, capsize=3, color='#4C72B0', alpha=0.85)
        for i, (m, c) in enumerate(zip(means, counts)):
            ax.text(i, m, f'{m:+.2f}\nn={c}', ha='center',
                    va='bottom' if m >= 0 else 'top', fontsize=7)
        overall = float(cos.mean())
        ax.axhline(overall, color='crimson', ls='--', lw=1.2,
                   label=f'pair mean {overall:+.3f}')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('mean cosine similarity')
        ax.set_ylim(min(-0.1, min(means) - 0.15), 1.0)
        ax.set_title(f'{pk}  ({sa} vs {sb})', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        print(f"[{pk}] {sa} vs {sb}:")
        for lbl, m, s, c in zip(labels, means, stds, counts):
            print(f"  {lbl}: cos={m:+.4f} +/- {s:.4f}  (n={c})")
        print(f"  -> pair mean {overall:+.4f}")

    fig.suptitle(f'{args.dataset_name} — per-episode matched-pair latent '
                 f"similarity\n{os.path.basename(args.checkpoint_dir.rstrip('/'))}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
