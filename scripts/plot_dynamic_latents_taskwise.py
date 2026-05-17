"""
Task-wise dynamic-latent analysis for cross_modal_vae3 checkpoints.

For each checkpoint, encodes the unpaired dynamic data of each object through
all three sensor encoders, then asks: **is the manipulation task (cylinder vs
plug) separable in the shared latent space?**

Two outputs per checkpoint:

  1. Linear separability (the "good separator" question, measured properly).
     A logistic-regression classifier cylinder-vs-plug is fit on the RAW μ
     vectors (the actual latent space — NOT t-SNE coords, whose distances are
     distorted). Reported as 5-fold cross-validated accuracy:
        - pooled  : all D+E+F latents together (shared latent space)
        - per-sensor: D-only, E-only, F-only
     ~0.5 → no separator exists; ~1.0 → tasks are cleanly linearly separable.

  2. t-SNE scatter, coloured by TASK ONLY (2 colours):
        cylinder = blue,  plug = red
     Sensor is shown by marker (D=o E=^ F=s) so you can still see the sensors,
     but colour now answers one question: do the two tasks split?
     Points the best in-sample linear separator gets WRONG are ringed in
     black — so you see *where* the separator fails.

One subplot per checkpoint (each has its own latent space → independent t-SNE).

Usage:
    python scripts/plot_dynamic_latents_taskwise.py \
        --checkpoint_dir \
            checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_dynalign_plain_cyl_plug_20260513_153826 \
            checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_pdsweep_pd8_20260516_044504 \
            checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_pdsweep_pd16_20260516_023747 \
        --device cuda:0 \
        --out dynamic_latents_taskwise.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch
import yaml
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.data.datasets import (
    DynamicSelfDataset, _build_sensor_transforms,
)
from joint_embedding_learning.utils.models_setup import get_model


_SENSOR_NAME = {'D': 'daimon', 'E': 'eflesh', 'F': 'flexi'}

# Task → colour (just two). Sensor → marker shape.
TASK_COLOR = {'cylinder': '#2171b5', 'plug': '#e6261f'}   # blue / red
MARKERS = {'D': 'o', 'E': '^', 'F': 's'}


def _mmd_rbf(x, y, bandwidths=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """Multi-kernel RBF MMD² between two sample sets (numpy in, float out)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    out = 0.0
    for s in bandwidths:
        out += (torch.exp(-xx / (2 * s)).mean()
                + torch.exp(-yy / (2 * s)).mean()
                - 2 * torch.exp(-xy / (2 * s)).mean()).item()
    return out / len(bandwidths)


def _lin_sep_acc(X, y, seed):
    """5-fold CV accuracy of a linear (logistic-reg) cylinder-vs-plug classifier.

    Measured in the RAW latent space. Needs both classes present.
    """
    if len(np.unique(y)) < 2:
        return float('nan')
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    n_per_class = np.bincount(y).min()
    folds = int(min(5, n_per_class))
    if folds < 2:
        return float('nan')
    scores = cross_val_score(clf, X, y, cv=folds, scoring='accuracy')
    return float(scores.mean())


@torch.no_grad()
def _encode_object_sensor(model, dyn_root, obj, sensor, dyn_details,
                           image_size, device, max_points, batch_size, seed):
    """Return (N, latent_dim) numpy μ for one object × one sensor of dynamic data."""
    sensor_name = _SENSOR_NAME[sensor]
    sensor_root = os.path.join(dyn_root, obj, sensor_name)
    if not os.path.isdir(sensor_root):
        print(f"  [skip] missing {sensor_root}")
        return None

    transform, mean, std = _build_sensor_transforms(dyn_details, sensor_name, image_size)
    ds = DynamicSelfDataset(sensor_root, sensor_name,
                            transform=transform, mean=mean, std=std, device='cpu')
    if len(ds) == 0:
        print(f"  [skip] empty {sensor_root}")
        return None

    n = len(ds)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(n, size=max_points, replace=False).tolist())
        ds = torch.utils.data.Subset(ds, idx)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    mus = []
    for x, _ in loader:
        x = x.to(device)
        _, mu, _, _ = model._enc(x, sensor)
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0).astype(np.float32)


def _load_model(checkpoint_dir, device):
    with open(os.path.join(checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model('cross_modal_vae3', model_details).to(device)
    ckpt = torch.load(os.path.join(checkpoint_dir, 'model.pt'),
                      map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    return model, model_details


def process_checkpoint(ax, checkpoint_dir, dyn_root, dyn_details, objects,
                        device, max_points, batch_size, tsne_perplexity, seed):
    """Encode, measure linear separability, t-SNE scatter. Returns metrics dict."""
    name = os.path.basename(checkpoint_dir.rstrip('/'))
    print(f"\n=== {name} ===")
    model, model_details = _load_model(checkpoint_dir, device)
    image_size = dyn_details.get('image_size', 224)
    pd = model_details.get('projection_dim', '?')

    # Encode every (object, sensor) group.
    groups = {}   # (obj, sensor) -> (N, d)
    for obj in objects:
        for sensor in ('D', 'E', 'F'):
            mu = _encode_object_sensor(model, dyn_root, obj, sensor, dyn_details,
                                       image_size, device, max_points,
                                       batch_size, seed)
            if mu is not None and len(mu) > 0:
                groups[(obj, sensor)] = mu
                print(f"  {obj:10s} {sensor}: {mu.shape}")

    if not groups:
        ax.set_title(f"{name}\n(no data)")
        return {}

    # ----- pooled arrays: X (latents), y (0=cyl 1=plug), sensor per point -----
    keys = list(groups.keys())
    sizes = [len(groups[k]) for k in keys]
    X = np.concatenate([groups[k] for k in keys], axis=0)
    y = np.concatenate([
        np.full(len(groups[k]), 0 if k[0] == 'cylinder' else 1, dtype=int)
        for k in keys
    ])
    sensor_of = np.concatenate([
        np.array([k[1]] * len(groups[k])) for k in keys
    ])

    # ----- linear separability (the "good separator" question) -----
    acc = {'pooled': _lin_sep_acc(X, y, seed)}
    for s in ('D', 'E', 'F'):
        m = sensor_of == s
        acc[s] = _lin_sep_acc(X[m], y[m], seed) if m.sum() > 0 else float('nan')

    # In-sample best separator → which points it gets wrong (for the overlay).
    wrong = np.zeros(len(y), dtype=bool)
    if len(np.unique(y)) == 2:
        clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(X, y)
        wrong = clf.predict(X) != y

    # Per-sensor MMD²(cyl, plug) — complementary distributional measure.
    mmd = {}
    for s in ('D', 'E', 'F'):
        if ('cylinder', s) in groups and ('plug', s) in groups:
            mmd[s] = _mmd_rbf(groups[('cylinder', s)], groups[('plug', s)])

    # ----- t-SNE (visual only) -----
    perp = min(tsne_perplexity, max(5, (len(X) - 1) // 3))
    emb = TSNE(n_components=2, perplexity=perp, init='pca',
               random_state=seed).fit_transform(X)

    # ----- scatter: colour by task, marker by sensor, ring misclassified -----
    off = 0
    for k, sz in zip(keys, sizes):
        obj, sensor = k
        sl = slice(off, off + sz)
        off += sz
        pts = emb[sl]
        w = wrong[sl]
        # correct points: no edge
        ax.scatter(pts[~w, 0], pts[~w, 1], s=9, alpha=0.5,
                   c=TASK_COLOR[obj], marker=MARKERS[sensor], edgecolors='none')
        # misclassified: black ring
        if w.any():
            ax.scatter(pts[w, 0], pts[w, 1], s=14, alpha=0.9,
                       c=TASK_COLOR[obj], marker=MARKERS[sensor],
                       edgecolors='black', linewidths=0.7)

    acc_str = (f"pooled={acc['pooled']:.3f}  "
               f"D={acc['D']:.3f} E={acc['E']:.3f} F={acc['F']:.3f}")
    ax.set_title(f"{name}\nprojection_dim={pd}   "
                 f"linear cyl/plug acc: {acc_str}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    return {'acc': acc, 'mmd': mmd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', nargs='+', required=True,
                        help='One or more checkpoint dirs (one subplot each).')
    parser.add_argument('--dynamic_root', type=str,
                        default='latent_data_processed_object/dynamic_data',
                        help='Dir with <object>/<sensor>/*.pt dynamic data.')
    parser.add_argument('--stats_dataset_name', type=str,
                        default='dataset_cross_modal_tri_dynamic_all',
                        help='Entry in datasets_config.yaml for dynamic '
                             'normalization stats + transforms.')
    parser.add_argument('--objects', nargs='+', default=['cylinder', 'plug'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_points', type=int, default=800,
                        help='Cap per (object, sensor) group.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tsne_perplexity', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='dynamic_latents_taskwise.png')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dyn_root = (args.dynamic_root if os.path.isabs(args.dynamic_root)
                else os.path.join(repo_root, args.dynamic_root))

    with open(os.path.join(repo_root, 'config', 'datasets_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    if args.stats_dataset_name not in cfg:
        raise KeyError(f"{args.stats_dataset_name!r} not in datasets_config.yaml")
    dyn_details = cfg[args.stats_dataset_name]

    n = len(args.checkpoint_dir)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.2), squeeze=False)
    axes = axes[0]

    results = {}
    for ax, ck in zip(axes, args.checkpoint_dir):
        results[os.path.basename(ck.rstrip('/'))] = process_checkpoint(
            ax, ck, dyn_root, dyn_details, args.objects, args.device,
            args.max_points, args.batch_size, args.tsne_perplexity, args.seed,
        )

    # Shared legend: 2 task colours + 3 sensor markers + misclassified ring.
    handles = [
        Patch(facecolor=TASK_COLOR['cylinder'], label='cylinder'),
        Patch(facecolor=TASK_COLOR['plug'], label='plug'),
        Line2D([], [], color='gray', marker='o', ls='', label='D (daimon)'),
        Line2D([], [], color='gray', marker='^', ls='', label='E (eflesh)'),
        Line2D([], [], color='gray', marker='s', ls='', label='F (flexi)'),
        Line2D([], [], color='gray', marker='o', ls='', markeredgecolor='black',
               markeredgewidth=1.0, label='misclassified by linear sep.'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8,
               frameon=False)
    fig.suptitle('Dynamic-data latents — task separability (cylinder vs plug)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    fig.savefig(out, dpi=150)
    print(f"\nSaved → {out}")

    print("\nLinear cylinder-vs-plug separability (5-fold CV accuracy on raw μ):")
    print("  ~0.50 = no separator   ~1.00 = cleanly linearly separable")
    for name, r in results.items():
        if not r:
            continue
        a = r['acc']
        print(f"  {name}")
        print(f"    pooled={a['pooled']:.3f}  D={a['D']:.3f}  "
              f"E={a['E']:.3f}  F={a['F']:.3f}")


if __name__ == '__main__':
    main()
