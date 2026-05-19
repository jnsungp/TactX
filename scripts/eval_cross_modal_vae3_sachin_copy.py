"""
Evaluation for cross_modal_vae3 checkpoints trained via the sachin
dynamic-align path. Mirrors eval_cross_modal_vae_tri.ipynb but adds the
crucial dynamic-data evaluation (the actual reason for the alignment
losses) and uses the per-pair-transform loader from
joint_embedding_learning.data.datasets_sachin.

What it produces (under --out_dir):
    cosine_dist_val.png             — pos vs neg histograms per pair (DE/EF/FD + DF transitive)
    tsne_paired_train.png           — t-SNE of paired-static train μ (modality + pair links)
    tsne_paired_val.png             — same on val
    tsne_dynamic.png                — **t-SNE of dynamic μ alone** (the goal: 3 clusters should merge)
    tsne_static_vs_dynamic.png      — combined: do dynamic latents land in the static manifold?
    mmd_static_vs_dynamic.txt       — numeric MMD per modality (lower = better domain match)
    retrieval.txt                   — Top-1/5/10 cross-modal retrieval on paired val
    recon_daimon.png, recon_eflesh.png, recon_flexi.png — reconstruction sanity checks

Usage:
python scripts/eval_cross_modal_vae3_sachin_copy.py \
    --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_dynamic_paired_run_dynamic_paired_efs2_20260519_043717 \
    --dynamic_data_root latent_data_processed_dynamic/pnp latent_data_processed_dynamic/wipe latent_data_processed_dynamic/plug \
    --dynamic_objects pnp wipe plug \
    --device cuda:0

Optional:
    --objects cylinder            # static paired objects (default: cylinder)
    --dataset_name dataset_sachin # which sachin entry to load
    --max_tsne_points 3000        # cap per-modality points fed to t-SNE for speed
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.data.datasets_sachin import (
    get_paired_tri_datasets_sachin,
    get_cross_modal_tri_datasets_sachin,
)
from joint_embedding_learning.utils.models_setup import get_model


_SENSORS = ('D', 'E', 'F')
_PAIRS = ('de', 'ef', 'fd')


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_pair_mu(dataset, pair_name, model, device, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    mu_a, mu_b = [], []
    for (xa, xb), _ in tqdm(loader, desc=f"paired_{pair_name}"):
        out = model.forward_pair(xa.to(device), xb.to(device), pair_name)
        mu_a.append(out['mu_a'].cpu())
        mu_b.append(out['mu_b'].cpu())
    return torch.cat(mu_a), torch.cat(mu_b)


@torch.no_grad()
def extract_self_mu(dataset, sensor, model, device, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    mus = []
    for x, _ in tqdm(loader, desc=f"dynamic_{sensor}"):
        out = model.forward_self(x.to(device), sensor)
        mus.append(out['mu'].cpu())
    return torch.cat(mus)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_pos_neg(mu_a, mu_b, sample_neg=5000):
    a = F.normalize(mu_a, dim=1)
    b = F.normalize(mu_b, dim=1)
    pos = (a * b).sum(dim=1).cpu().numpy()
    N = a.shape[0]
    idx_a = np.random.randint(0, N, sample_neg)
    idx_b = np.random.randint(0, N, sample_neg)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N
    neg = (a[idx_a] * b[idx_b]).sum(dim=1).cpu().numpy()
    return pos, neg


def transitive_df_cosine(mu_D_de, mu_E_de, mu_E_ef, mu_F_ef, sample_neg=5000):
    e_de_n = F.normalize(mu_E_de, dim=1)
    e_ef_n = F.normalize(mu_E_ef, dim=1)
    nn_idx = (e_ef_n @ e_de_n.t()).argmax(dim=1)
    mu_D_match = mu_D_de[nn_idx]
    d_n = F.normalize(mu_D_match, dim=1)
    f_n = F.normalize(mu_F_ef, dim=1)
    pos = (d_n * f_n).sum(dim=1).cpu().numpy()
    N = d_n.shape[0]
    idx_a = np.random.randint(0, N, sample_neg)
    idx_b = np.random.randint(0, N, sample_neg)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N
    neg = (d_n[idx_a] * f_n[idx_b]).sum(dim=1).cpu().numpy()
    return pos, neg


def mmd_rbf_multi(x, y, bandwidths=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """Multi-bandwidth RBF MMD². Same formula as the training-time loss."""
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    mmd2 = 0.0
    for s in bandwidths:
        mmd2 += (torch.exp(-xx / (2 * s)).mean()
                 + torch.exp(-yy / (2 * s)).mean()
                 - 2 * torch.exp(-xy / (2 * s)).mean())
    return float(mmd2 / len(bandwidths))


def topk_retrieval(query, gallery, ks=(1, 5, 10)):
    """For each query[i], find rank of gallery[i] in similarity-sorted gallery.
    Returns dict {k: accuracy}."""
    q = F.normalize(query, dim=1)
    g = F.normalize(gallery, dim=1)
    sim = q @ g.t()                              # (N, N)
    N = sim.shape[0]
    # ranking: top-k indices per row
    out = {}
    for k in ks:
        topk = sim.topk(k=min(k, N), dim=1).indices    # (N, k)
        match = (topk == torch.arange(N).unsqueeze(1)).any(dim=1)
        out[k] = match.float().mean().item()
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_tsne(points, perplexity=100, seed=42, metric='cosine'):
    """
    t-SNE for contrastive latent visualization.

    - Lower early_exaggeration makes clusters less aggressively pushed apart.
    """
    kw = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        early_exaggeration=4,
        metric=metric,
        random_state=seed,
    )

    if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) >= (1, 0):
        kw['max_iter'] = 1500
    else:
        kw['n_iter'] = 1500

    coords = TSNE(**kw).fit_transform(points)

    # Center coordinates for cleaner plotting.
    coords = coords - coords.mean(axis=0, keepdims=True)

    return coords


def _subsample(arr, n_max, seed=0):
    if len(arr) <= n_max:
        return arr, np.arange(len(arr))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(arr), n_max, replace=False)
    return arr[idx], idx


def plot_cosine_dist(pairs_data, out_path):
    fig, axes = plt.subplots(1, len(pairs_data), figsize=(5 * len(pairs_data), 4), sharey=False)
    if len(pairs_data) == 1:
        axes = [axes]
    bins = np.linspace(-1, 1, 61)
    for ax, (label, pos, neg, color) in zip(axes, pairs_data):
        ax.hist(neg, bins=bins, alpha=0.55, color='gray', label='neg', density=True)
        ax.hist(pos, bins=bins, alpha=0.75, color=color, label='pos', density=True)
        ax.axvline(pos.mean(), color=color, ls='--', lw=1.5)
        ax.axvline(neg.mean(), color='gray', ls='--', lw=1.5)
        ax.set_title(f"{label}\ngap = {pos.mean() - neg.mean():.3f}")
        ax.set_xlabel('cosine similarity')
        ax.legend(fontsize=8)
    plt.suptitle('μ Cosine: positive vs negative', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_paired(mu_pairs, split_name, out_path, n_max=2000):
    """Paired-only t-SNE (modality + pair links).

    mu_pairs: dict {'de': (mu_D, mu_E), 'ef': (mu_E, mu_F), 'fd': (mu_F, mu_D)}.
    """
    blocks = []
    info = []  # (sensor, pair, count)
    for pk in _PAIRS:
        ma, mb = mu_pairs[pk]
        sa, sb = {'de': ('D', 'E'), 'ef': ('E', 'F'), 'fd': ('F', 'D')}[pk]
        a_arr = F.normalize(ma, dim=1).numpy()
        b_arr = F.normalize(mb, dim=1).numpy()
        if n_max > 0:
            a_arr, _ = _subsample(a_arr, n_max, seed=hash(pk + 'a') & 0xFFFF)
            b_arr, _ = _subsample(b_arr, n_max, seed=hash(pk + 'b') & 0xFFFF)
        info.append((sa, pk, len(a_arr)))
        info.append((sb, pk, len(b_arr)))
        blocks.append(a_arr)
        blocks.append(b_arr)

    combined = np.concatenate(blocks, axis=0)
    print(f"  [{split_name}] paired t-SNE on {len(combined)} points")
    coords = _make_tsne(combined)

    # split coords back per block
    off = 0
    block_coords = []
    for (_, _, n) in info:
        block_coords.append(coords[off:off + n])
        off += n

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sensor_color = {'D': 'tab:blue', 'E': 'tab:orange', 'F': 'tab:green'}

    # Left: color by modality
    ax = axes[0]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=sensor_color[sensor],
                   alpha=0.4, s=12, label=sensor)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=s)
               for s, c in sensor_color.items()]
    ax.legend(handles=handles, fontsize=10)
    ax.set_title(f't-SNE paired [{split_name}] — by modality')
    ax.set_xticks([]); ax.set_yticks([])

    # Right: pair links
    ax = axes[1]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=sensor_color[sensor], alpha=0.3, s=10)
    # draw lines between paired a/b inside each pair
    link_colors = {'de': 'mediumpurple', 'ef': 'gray', 'fd': 'lightcoral'}
    rng = np.random.RandomState(0)
    for i, pk in enumerate(_PAIRS):
        ai = 2 * i
        bi = 2 * i + 1
        pa = block_coords[ai]
        pb = block_coords[bi]
        n = min(len(pa), len(pb), 500)
        idx = rng.choice(min(len(pa), len(pb)), n, replace=False)
        for k in idx:
            ax.plot([pa[k, 0], pb[k, 0]], [pa[k, 1], pb[k, 1]],
                    c=link_colors[pk], alpha=0.15, lw=0.5)
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=f'{pk} link')
               for pk, c in link_colors.items()]
    ax.legend(handles=handles, fontsize=10)
    ax.set_title(f't-SNE paired [{split_name}] — pair links')
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_dynamic(dyn_mus, out_path, n_max=2000, normalize=True,
                      perplexity=None):
    """t-SNE of dynamic-only μ across the three sensors.

    Success = D/E/F clouds overlap in the shared latent space.
    """
    blocks = []
    sensor_color = {
        'D': 'tab:blue',
        'E': 'tab:orange',
        'F': 'tab:green',
    }
    info = []

    for s in _SENSORS:
        arr = dyn_mus[s]
        if normalize:
            arr = F.normalize(arr, dim=1)
        arr = arr.numpy()
        if n_max > 0:
            arr, _ = _subsample(arr, n_max, seed=hash('dyn_' + s) & 0xFFFF)
        blocks.append(arr)
        info.append((s, len(arr)))

    combined = np.concatenate(blocks, axis=0)
    n_total = len(combined)
    # Perplexity must be < n_samples/3; auto-scale for small datasets
    if perplexity is None:
        perplexity = min(50, max(5, n_total // 6))
    print(f"  [dynamic] t-SNE on {n_total} points  "
          f"(perplexity={perplexity}, normalize={normalize})")

    metric = 'cosine' if normalize else 'euclidean'
    coords = _make_tsne(combined, perplexity=perplexity, seed=42,
                        metric=metric)

    fig, ax = plt.subplots(figsize=(7.0, 6.5))

    off = 0
    for (s, n) in info:
        pts = coords[off:off + n]
        off += n

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=sensor_color[s],
            alpha=0.2,
            s=24,
            label=s,
            linewidths=0,
            rasterized=True,
        )

    ax.set_title('t-SNE of dynamic latent μ', fontsize=15, pad=10)

    ax.set_xticks([])
    ax.set_yticks([])

    # Remove box boundary for paper-style visualization.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Keep aspect ratio stable.
    ax.set_aspect('equal', adjustable='box')

    # Add a small margin so points are not clipped.
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_pad = 0.06 * (x_max - x_min + 1e-8)
    y_pad = 0.06 * (y_max - y_min + 1e-8)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.legend(
        fontsize=11,
        frameon=False,
        loc='upper right',
        markerscale=1.2,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_dynamic_bytask(dyn_mus, task_labels, out_path, n_max=2000,
                             normalize=True, perplexity=None):
    """t-SNE colored by task (e.g. pnp vs wipe) instead of modality.

    task_labels: dict {sensor: list_of_task_names} same length as dyn_mus[sensor].
    """
    import matplotlib.cm as cm

    blocks = []
    task_per_point = []
    sensor_per_point = []

    for s in _SENSORS:
        arr = dyn_mus[s]
        if normalize:
            arr = F.normalize(arr, dim=1)
        arr = arr.numpy()
        labels = np.array(task_labels[s])
        if n_max > 0 and len(arr) > n_max:
            rng = np.random.RandomState(hash('task_' + s) & 0xFFFF)
            idx = rng.choice(len(arr), n_max, replace=False)
            arr = arr[idx]
            labels = labels[idx]
        blocks.append(arr)
        task_per_point.extend(labels.tolist())
        sensor_per_point.extend([s] * len(arr))

    combined = np.concatenate(blocks, axis=0)
    n_total = len(combined)
    if perplexity is None:
        perplexity = min(50, max(5, n_total // 6))
    print(f"  [dynamic_bytask] t-SNE on {n_total} points  (perplexity={perplexity})")

    metric = 'cosine' if normalize else 'euclidean'
    coords = _make_tsne(combined, perplexity=perplexity, seed=42, metric=metric)

    task_per_point = np.array(task_per_point)
    sensor_per_point = np.array(sensor_per_point)
    unique_tasks = sorted(set(task_per_point))
    # Distinct colors for tasks
    task_cmap = {t: c for t, c in zip(unique_tasks,
                 ['tab:red', 'tab:cyan', 'tab:purple', 'tab:olive',
                  'tab:pink', 'tab:brown'][:len(unique_tasks)])}
    sensor_marker = {'D': 'o', 'E': '^', 'F': 's'}

    fig, ax = plt.subplots(figsize=(8, 7))
    for task in unique_tasks:
        for s in _SENSORS:
            mask = (task_per_point == task) & (sensor_per_point == s)
            if not mask.any():
                continue
            pts = coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=task_cmap[task], marker=sensor_marker[s],
                       alpha=0.5, s=20, linewidths=0, rasterized=True,
                       label=f'{task} ({s})')

    ax.set_title('t-SNE dynamic μ — colored by task', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Legend: task colors + sensor shapes
    from matplotlib.lines import Line2D
    handles = []
    for t, c in task_cmap.items():
        handles.append(Line2D([], [], marker='o', ls='', color=c,
                              markersize=8, label=t))
    for s, m in sensor_marker.items():
        handles.append(Line2D([], [], marker=m, ls='', color='gray',
                              markersize=8, label=s))
    ax.legend(handles=handles, fontsize=9, frameon=False, loc='upper right',
              ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_dynamic_paired(dyn_pair_mus, out_path, n_max=2000, normalize=True,
                             perplexity=None):
    """t-SNE of dynamic μ with pair links drawn between matched frames.

    dyn_pair_mus: dict {'de': (mu_D, mu_E), 'ef': (mu_E, mu_F), 'fd': (mu_F, mu_D)}
    """
    sensor_color = {'D': 'tab:blue', 'E': 'tab:orange', 'F': 'tab:green'}
    link_color = {'de': 'mediumpurple', 'ef': 'gray', 'fd': 'lightcoral'}

    blocks = []
    info = []  # (sensor, pair_key, n_points)
    for pk in _PAIRS:
        ma, mb = dyn_pair_mus[pk]
        sa, sb = {'de': ('D', 'E'), 'ef': ('E', 'F'), 'fd': ('F', 'D')}[pk]
        a_arr = F.normalize(ma, dim=1).numpy() if normalize else ma.numpy()
        b_arr = F.normalize(mb, dim=1).numpy() if normalize else mb.numpy()
        # Subsample both sides identically to preserve pairing
        if n_max > 0 and len(a_arr) > n_max:
            rng = np.random.RandomState(hash(pk) & 0xFFFF)
            idx = rng.choice(len(a_arr), n_max, replace=False)
            a_arr = a_arr[idx]
            b_arr = b_arr[idx]
        info.append((sa, pk, len(a_arr)))
        info.append((sb, pk, len(b_arr)))
        blocks.append(a_arr)
        blocks.append(b_arr)

    combined = np.concatenate(blocks, axis=0)
    n_total = len(combined)
    if perplexity is None:
        perplexity = min(50, max(5, n_total // 6))
    print(f"  [dynamic_paired] t-SNE on {n_total} points  (perplexity={perplexity})")

    metric = 'cosine' if normalize else 'euclidean'
    coords = _make_tsne(combined, perplexity=perplexity, seed=42, metric=metric)

    # Split coords back
    off = 0
    block_coords = []
    for (_, _, n) in info:
        block_coords.append(coords[off:off + n])
        off += n

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Left: color by modality
    ax = axes[0]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=sensor_color[sensor],
                   alpha=0.5, s=20, linewidths=0, rasterized=True)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=s, markersize=8)
               for s, c in sensor_color.items()]
    ax.legend(handles=handles, fontsize=10, frameon=False, loc='upper right')
    ax.set_title('Dynamic μ — by modality', fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Right: pair links
    ax = axes[1]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=sensor_color[sensor],
                   alpha=0.3, s=14, linewidths=0, rasterized=True)
    # Draw lines between paired a/b within each pair
    rng = np.random.RandomState(0)
    for i, pk in enumerate(_PAIRS):
        ai = 2 * i
        bi = 2 * i + 1
        pa = block_coords[ai]
        pb = block_coords[bi]
        n_links = min(len(pa), len(pb), 500)
        idx = rng.choice(min(len(pa), len(pb)), n_links, replace=False)
        for k in idx:
            ax.plot([pa[k, 0], pb[k, 0]], [pa[k, 1], pb[k, 1]],
                    c=link_color[pk], alpha=0.2, lw=0.5)
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=f'{pk} link')
               for pk, c in link_color.items()]
    ax.legend(handles=handles, fontsize=10, frameon=False, loc='upper right')
    ax.set_title('Dynamic μ — pair links', fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_static_vs_dynamic(static_mu_per_sensor, dyn_mus, out_path, n_max=1500):
    """Combined t-SNE: do dynamic latents land in the static manifold per modality?

    Plots 6 clouds (D-static, D-dynamic, E-static, E-dynamic, F-static, F-dynamic).
    """
    sensor_color = {'D': 'tab:blue', 'E': 'tab:orange', 'F': 'tab:green'}
    blocks = []
    info = []
    for s in _SENSORS:
        st = F.normalize(static_mu_per_sensor[s], dim=1).numpy()
        dy = F.normalize(dyn_mus[s], dim=1).numpy()
        if n_max > 0:
            st, _ = _subsample(st, n_max, seed=hash(f'st_{s}') & 0xFFFF)
            dy, _ = _subsample(dy, n_max, seed=hash(f'dy_{s}') & 0xFFFF)
        blocks.append(st); info.append((s, 'static', len(st)))
        blocks.append(dy); info.append((s, 'dynamic', len(dy)))

    combined = np.concatenate(blocks, axis=0)
    print(f"  [static+dynamic] t-SNE on {len(combined)} points")
    coords = _make_tsne(combined)

    fig, ax = plt.subplots(figsize=(9, 8))
    off = 0
    for (s, regime, n) in info:
        pts = coords[off:off + n]
        off += n
        marker = 'o' if regime == 'static' else 'x'
        alpha = 0.4 if regime == 'static' else 0.6
        ax.scatter(pts[:, 0], pts[:, 1], c=sensor_color[s], alpha=alpha, s=14,
                   marker=marker, label=f"{s} {regime}")
    ax.set_title('t-SNE: static (circles) vs dynamic (x) per modality\n'
                 'Success = same-color circles and x overlap')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Reconstruction visualizations (Daimon, eFlesh, Flexi)
# ---------------------------------------------------------------------------

@torch.no_grad()
def viz_recon_daimon(val_de_dataset, model, device, daimon_stats, out_path, n=8):
    loader = DataLoader(val_de_dataset, batch_size=n, shuffle=True, drop_last=True)
    (xa, xb), _ = next(iter(loader))
    out = model.forward_pair(xa.to(device), xb.to(device), 'de')

    def to_vis(x):
        x = x.cpu().float()
        mn, mx = x.min(), x.max()
        return ((x - mn) / (mx - mn + 1e-8)).clamp(0, 1)

    rows = [
        ('GT Daimon', [to_vis(xa[i]) for i in range(n)]),
        ('Self D(z_D)', [to_vis(out['recon_aa'][i].cpu()) for i in range(n)]),
        ('Cross D(z_E)', [to_vis(out['recon_ba'][i].cpu()) for i in range(n)]),
    ]
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    for r, (label, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            if img.shape[0] == 3:
                ax.imshow(img.permute(1, 2, 0).numpy())
            else:
                ax.imshow(img.squeeze().numpy(), cmap='gray')
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    plt.suptitle('Daimon recon (de_pair val)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


@torch.no_grad()
def viz_recon_eflesh(val_de_dataset, model, device, out_path, n=8):
    loader = DataLoader(val_de_dataset, batch_size=n, shuffle=True, drop_last=True)
    (xa, xb), _ = next(iter(loader))
    out = model.forward_pair(xa.to(device), xb.to(device), 'de')
    gt = xb.cpu().numpy()
    self_r = out['recon_bb'].cpu().numpy()
    cross_r = out['recon_ab'].cpu().numpy()

    fig, axes = plt.subplots(n, 3, figsize=(10, n * 1.2))
    for i in range(n):
        for j, (arr, ttl) in enumerate([
            (gt[i], 'GT'),
            (self_r[i], 'Self D_E(z_E)'),
            (cross_r[i], 'Cross D_E(z_D)'),
        ]):
            axes[i, j].bar(range(15), arr, color='steelblue', alpha=0.8)
            if i == 0:
                axes[i, j].set_title(ttl, fontsize=8)
            axes[i, j].tick_params(labelsize=6)
    plt.suptitle('eFlesh recon (de_pair val, z-scored)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


@torch.no_grad()
def viz_recon_flexi(val_ef_dataset, model, device, out_path, n=8):
    loader = DataLoader(val_ef_dataset, batch_size=n, shuffle=True, drop_last=True)
    (xe, xf), _ = next(iter(loader))
    out = model.forward_pair(xe.to(device), xf.to(device), 'ef')
    gt = xf.cpu().squeeze(1) if xf.dim() == 4 else xf.cpu()
    self_r = out['recon_bb'].cpu().squeeze(1) if out['recon_bb'].dim() == 4 else out['recon_bb'].cpu()
    cross_r = out['recon_ab'].cpu().squeeze(1) if out['recon_ab'].dim() == 4 else out['recon_ab'].cpu()

    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    for c in range(n):
        for r, (arr, ttl) in enumerate([
            (gt[c].numpy(), 'GT FlexiTac'),
            (self_r[c].numpy(), 'Self D_F(z_F)'),
            (cross_r[c].numpy(), 'Cross D_F(z_E)'),
        ]):
            ax = axes[r, c]
            ax.imshow(arr, cmap='viridis')
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(ttl, fontsize=8)
    plt.suptitle('FlexiTac recon (ef_pair val)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True,
                        help='checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_<name>_<TS>')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset_name', default='dataset_sachin')
    parser.add_argument('--dynamic_dataset_name', default='dataset_cross_modal_tri_dynamic')
    parser.add_argument('--dynamic_data_root', nargs='+', default=None,
                        help='One or more paired-layout dynamic directories '
                             '(each must contain de_pair/, ef_pair/, fd_pair/). '
                             'e.g. --dynamic_data_root latent_data_processed_dynamic/pnp '
                             'latent_data_processed_dynamic/wipe')
    parser.add_argument('--dynamic_objects', nargs='+', default=None,
                        help='Object sub-dir names corresponding 1:1 to each '
                             '--dynamic_data_root entry. '
                             'e.g. --dynamic_objects pnp wipe')
    parser.add_argument('--objects', nargs='+', default=['cylinder'])
    parser.add_argument('--out_dir', default=None,
                        help='Default: <checkpoint_dir>/eval_sachin_cp/')
    parser.add_argument('--max_tsne_points', type=int, default=0,
                        help='Cap per-modality points for t-SNE. 0 = use ALL points.')
    parser.add_argument('--no_l2_norm', action='store_true',
                        help='Skip L2 normalization before t-SNE (use raw μ, '
                             'euclidean metric). Helps when data is sparse.')
    parser.add_argument('--tsne_perplexity', type=int, default=None,
                        help='Override t-SNE perplexity (default: auto-scale '
                             'based on data size).')
    parser.add_argument('--max_mmd_points', type=int, default=2000,
                        help='Sample size per side for MMD numeric metric.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--daimon_deform_only', action='store_true',
                        help='Encode Daimon as channel-0 replicated 3x '
                             '(deform-only). Applies to BOTH paired and '
                             'dynamic data. Use when evaluating a '
                             'dataset_sachin_static_deform checkpoint.')
    parser.add_argument('--ef_stride', type=int, default=1,
                        help='Per-episode frame stride for ef_pair — set to '
                             'MATCH the training run (e.g. 3) so the val set '
                             'is the genuine held-out partition and metrics '
                             'are comparable.')
    parser.add_argument('--de_stride', type=int, default=1,
                        help='Per-episode frame stride for de_pair (match training).')
    parser.add_argument('--fd_stride', type=int, default=1,
                        help='Per-episode frame stride for fd_pair (match training).')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'eval_sachin_wipe_pnp')
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load model ──
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model('cross_modal_vae3', model_details).to(args.device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'model.pt'),
                      map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded {args.checkpoint_dir}/model.pt")

    # Optional: force Daimon to deform-only (ch0 replicated 3x) at encode time.
    # _enc is the single chokepoint for forward_pair AND forward_self, so this
    # one patch covers paired + dynamic + recon-viz Daimon uniformly.
    # Idempotent — ch0 of an already-[ch0,ch0,ch0] tensor is ch0 — so it is
    # safe even when paired data already came from a *_deform dataset.
    if args.daimon_deform_only:
        _orig_enc = model._enc

        def _enc_deform(x, sensor, _orig=_orig_enc):
            if sensor == 'D' and hasattr(x, 'dim') and x.dim() == 4:
                x = x[:, 0:1].repeat(1, 3, 1, 1)
            return _orig(x, sensor)

        model._enc = _enc_deform
        print("[daimon_deform_only] Daimon encoded as ch0 replicated 3x "
              "(paired + dynamic)")

    # ── Load paired (sachin) + dynamic datasets ──
    # Per-pair strides MUST match the training run (e.g. --ef_stride 3) so the
    # random_split val set is the genuine held-out partition.
    _pair_strides = {k: v for k, v in
                     (('de', args.de_stride), ('ef', args.ef_stride),
                      ('fd', args.fd_stride)) if v > 1}
    if _pair_strides:
        print(f"Per-pair frame strides: {_pair_strides}")
    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device='cpu', objects=args.objects,
        pair_strides=_pair_strides or None,
    )
    print('Paired val sizes:', {k: len(v) for k, v in paired_val.items()})

    # ── Extract paired μ (train + val) ──
    print('Extracting paired-static μ...')
    mu_pairs_tr, mu_pairs_v = {}, {}
    for pk in _PAIRS:
        mu_pairs_tr[pk] = extract_pair_mu(paired_train[pk], pk, model, args.device)
        mu_pairs_v[pk] = extract_pair_mu(paired_val[pk], pk, model, args.device)

    # ── Load + extract dynamic μ ──
    print('Extracting dynamic μ...')
    _dpm = None  # will hold per-pair mus if dynamic_data_root is used
    # task_labels[sensor] = list of task name per sample (for multi-root coloring)
    _task_labels = None
    if args.dynamic_data_root:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sachin_cfg_path = os.path.join(repo_root, 'config', 'datasets_config_sachin.yaml')
        with open(sachin_cfg_path) as f:
            sachin_cfg = yaml.safe_load(f)

        # Build object list per root (1:1 correspondence)
        dyn_roots = args.dynamic_data_root
        if args.dynamic_objects:
            if len(args.dynamic_objects) != len(dyn_roots):
                raise ValueError(
                    f"--dynamic_objects ({len(args.dynamic_objects)}) must match "
                    f"--dynamic_data_root ({len(dyn_roots)}) in length.")
            dyn_objs_per_root = [[o] for o in args.dynamic_objects]
        else:
            dyn_objs_per_root = [args.objects for _ in dyn_roots]

        # Extract from each root and accumulate
        all_dpm = {pk: ([], []) for pk in _PAIRS}  # pk -> ([mu_a...], [mu_b...])
        _task_labels = {s: [] for s in _SENSORS}
        for root_i, (droot, dobjs) in enumerate(zip(dyn_roots, dyn_objs_per_root)):
            task_name = dobjs[0] if len(dobjs) == 1 else os.path.basename(droot)
            dyn_details = dict(sachin_cfg[args.dataset_name])
            dyn_details['data_root'] = droot
            dyn_pairs_tr_i, _, _ = get_cross_modal_tri_datasets_sachin(
                args.dataset_name, dyn_details, device='cpu', objects=dobjs,
            )
            print(f'  [{task_name}] sizes: {{{", ".join(f"{k}: {len(v)}" for k,v in dyn_pairs_tr_i.items())}}}')
            for pk in _PAIRS:
                mu_a, mu_b = extract_pair_mu(dyn_pairs_tr_i[pk], pk, model, args.device)
                all_dpm[pk][0].append(mu_a)
                all_dpm[pk][1].append(mu_b)
                sa, sb = {'de': ('D', 'E'), 'ef': ('E', 'F'), 'fd': ('F', 'D')}[pk]
                _task_labels[sa].extend([task_name] * len(mu_a))
                _task_labels[sb].extend([task_name] * len(mu_b))

        # Concat per pair
        _dpm = {}
        for pk in _PAIRS:
            _dpm[pk] = (torch.cat(all_dpm[pk][0], dim=0),
                        torch.cat(all_dpm[pk][1], dim=0))
        dyn_mus = {
            'D': torch.cat([_dpm['de'][0], _dpm['fd'][1]], dim=0),
            'E': torch.cat([_dpm['de'][1], _dpm['ef'][0]], dim=0),
            'F': torch.cat([_dpm['ef'][1], _dpm['fd'][0]], dim=0),
        }
        print('Dynamic μ sizes:', {s: len(v) for s, v in dyn_mus.items()})
    else:
        dyn_train, _dv, _ = get_all_contrastive_datasets(
            args.dynamic_dataset_name, device='cpu',
        )
        print('Dynamic sizes:', {k: len(v) for k, v in dyn_train.items()})
        dyn_mus = {s: extract_self_mu(dyn_train[s], s, model, args.device)
                   for s in _SENSORS}

    # ── Aggregate static μ per modality (canonical source per sensor) ──
    # Same convention as the training step's MMD bridge.
    static_mu_per_sensor = {
        'D': mu_pairs_tr['de'][0],
        'E': mu_pairs_tr['de'][1],
        'F': mu_pairs_tr['ef'][1],
    }

    # ── Cosine similarity per pair (val) ──
    print('Computing cosine distributions...')
    pos_de, neg_de = cosine_pos_neg(*mu_pairs_v['de'])
    pos_ef, neg_ef = cosine_pos_neg(*mu_pairs_v['ef'])
    pos_fd, neg_fd = cosine_pos_neg(*mu_pairs_v['fd'])
    pos_df, neg_df = transitive_df_cosine(
        mu_pairs_v['de'][0], mu_pairs_v['de'][1],
        mu_pairs_v['ef'][0], mu_pairs_v['ef'][1],
    )
    pairs_data = [
        ('DE direct', pos_de, neg_de, '#4C72B0'),
        ('EF direct', pos_ef, neg_ef, '#DD8452'),
        ('FD direct', pos_fd, neg_fd, '#55A868'),
        ('DF transitive', pos_df, neg_df, '#C44E52'),
    ]
    plot_cosine_dist(pairs_data, os.path.join(out_dir, 'cosine_dist_val.png'))

    # ── MMD static vs dynamic per modality ──
    print('Computing MMD static-vs-dynamic...')
    mmd_lines = ['MMD² static vs dynamic per modality (lower = better):']
    for s in _SENSORS:
        st = static_mu_per_sensor[s]
        dy = dyn_mus[s]
        n = min(args.max_mmd_points, len(st), len(dy))
        rng = np.random.RandomState(args.seed)
        ids_s = rng.choice(len(st), n, replace=False)
        ids_d = rng.choice(len(dy), n, replace=False)
        mmd = mmd_rbf_multi(st[ids_s], dy[ids_d])
        mmd_lines.append(f"  {s}: MMD² = {mmd:.6f}  (n_static={len(st)}, n_dynamic={len(dy)})")
    mmd_txt = '\n'.join(mmd_lines)
    print(mmd_txt)
    with open(os.path.join(out_dir, 'mmd_static_vs_dynamic.txt'), 'w') as f:
        f.write(mmd_txt + '\n')

    # ── Cross-modal retrieval on paired val ──
    print('Computing retrieval Top-k...')
    retr_lines = ['Cross-modal retrieval (paired val, Top-1/5/10):']
    for pk in _PAIRS:
        ma, mb = mu_pairs_v[pk]
        ab = topk_retrieval(ma, mb)
        ba = topk_retrieval(mb, ma)
        retr_lines.append(f"  {pk} a→b: " + '  '.join(f"top{k}={v:.4f}" for k, v in ab.items()))
        retr_lines.append(f"  {pk} b→a: " + '  '.join(f"top{k}={v:.4f}" for k, v in ba.items()))
    retr_txt = '\n'.join(retr_lines)
    print(retr_txt)
    with open(os.path.join(out_dir, 'retrieval.txt'), 'w') as f:
        f.write(retr_txt + '\n')

    # ── t-SNE plots ──
    print('Computing t-SNE plots...')
    plot_tsne_paired(mu_pairs_tr, 'train', os.path.join(out_dir, 'tsne_paired_train.png'),
                     n_max=args.max_tsne_points)
    plot_tsne_paired(mu_pairs_v, 'val', os.path.join(out_dir, 'tsne_paired_val.png'),
                     n_max=args.max_tsne_points)
    plot_tsne_dynamic(dyn_mus, os.path.join(out_dir, 'tsne_dynamic.png'),
                      n_max=args.max_tsne_points,
                      normalize=not args.no_l2_norm,
                      perplexity=args.tsne_perplexity)
    if _dpm is not None:
        plot_tsne_dynamic_paired(
            _dpm, os.path.join(out_dir, 'tsne_dynamic_paired.png'),
            n_max=args.max_tsne_points,
            normalize=not args.no_l2_norm,
            perplexity=args.tsne_perplexity)
    if _task_labels is not None and len(set(sum((_task_labels[s] for s in _SENSORS), []))) > 1:
        plot_tsne_dynamic_bytask(
            dyn_mus, _task_labels,
            os.path.join(out_dir, 'tsne_dynamic_bytask.png'),
            n_max=args.max_tsne_points,
            normalize=not args.no_l2_norm,
            perplexity=args.tsne_perplexity)
    plot_tsne_static_vs_dynamic(
        static_mu_per_sensor, dyn_mus,
        os.path.join(out_dir, 'tsne_static_vs_dynamic.png'),
        n_max=min(args.max_tsne_points, 1500) if args.max_tsne_points > 0 else 0,
    )

    # ── Reconstruction visualizations ──
    print('Reconstruction visualizations...')
    # Look up daimon stats from dataset_sachin yaml (loaded inside the loader)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, 'config', 'datasets_config_sachin.yaml')) as f:
        dscfg = yaml.safe_load(f)[args.dataset_name]
    viz_recon_daimon(paired_val['de'], model, args.device,
                     dscfg.get('daimon_stats', None),
                     os.path.join(out_dir, 'recon_daimon.png'))
    viz_recon_eflesh(paired_val['de'], model, args.device,
                     os.path.join(out_dir, 'recon_eflesh.png'))
    viz_recon_flexi(paired_val['ef'], model, args.device,
                    os.path.join(out_dir, 'recon_flexi.png'))

    print(f"\nDone. All outputs under: {out_dir}/")


if __name__ == '__main__':
    main()