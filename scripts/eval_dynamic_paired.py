"""
Evaluation for cross_modal_vae3 trained on dynamic paired data
(dataset_sachin_dynamic_paired: ep0000-03 train, ep0004 val).

Outputs (under --out_dir):
    cosine_dist_val.png         — pos vs neg histograms per pair (val)
    tsne_train.png              — t-SNE of train μ (4 episodes, by modality + pair links)
    tsne_val.png                — t-SNE of val μ (ep0004 only)
    tsne_train_bytask.png       — t-SNE of train μ colored by task (plug/pnp/wipe)
    tsne_val_bytask.png         — t-SNE of val μ colored by task
    retrieval.txt               — Top-1/5/10 cross-modal retrieval on val
    recon_daimon.png            — Daimon reconstruction on val
    recon_eflesh.png            — eFlesh reconstruction on val
    recon_flexi.png             — FlexiTac reconstruction on val

Usage:
    python scripts/eval_dynamic_paired.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_dynamic_paired_run_dynamic_paired_efs2_20260519_043717 \
        --device cuda:0

Optional:
    --dataset_name dataset_sachin_dynamic_paired
    --objects plug pnp wipe
    --ef_stride 2          # match training stride
    --max_tsne_points 3000
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
from collections import defaultdict
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from joint_embedding_learning.data.datasets_sachin import (
    get_paired_tri_datasets_sachin,
)
from joint_embedding_learning.utils.models_setup import get_model


_SENSORS = ('D', 'E', 'F')
_PAIRS = ('de', 'ef', 'fd')
_PAIR_SENSORS = {'de': ('D', 'E'), 'ef': ('E', 'F'), 'fd': ('F', 'D')}


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


def _get_task_labels(dataset, pair_key):
    """Extract task (object) names per sample from a dataset or Subset."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = dataset.indices
        obj_ids = [base.object_per_sample[i] for i in indices]
        obj_names = base.object_names
    else:
        obj_ids = dataset.object_per_sample
        obj_names = dataset.object_names
    return [obj_names[oid] for oid in obj_ids]


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


def topk_retrieval(query, gallery, ks=(1, 5, 10)):
    q = F.normalize(query, dim=1)
    g = F.normalize(gallery, dim=1)
    sim = q @ g.t()
    N = sim.shape[0]
    out = {}
    for k in ks:
        topk = sim.topk(k=min(k, N), dim=1).indices
        match = (topk == torch.arange(N).unsqueeze(1)).any(dim=1)
        out[k] = match.float().mean().item()
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_tsne(points, perplexity=None, seed=42, metric='cosine'):
    n_total = len(points)
    if perplexity is None:
        perplexity = min(50, max(5, n_total // 6))

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
    plt.suptitle('μ Cosine: positive vs negative (dynamic val)', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_modality(mu_pairs, split_name, out_path, n_max=2000, normalize=True,
                       perplexity=None):
    """t-SNE colored by modality with pair links."""
    sensor_color = {'D': 'tab:blue', 'E': 'tab:orange', 'F': 'tab:green'}
    link_color = {'de': 'mediumpurple', 'ef': 'gray', 'fd': 'lightcoral'}

    blocks = []
    info = []
    for pk in _PAIRS:
        ma, mb = mu_pairs[pk]
        sa, sb = _PAIR_SENSORS[pk]
        a_arr = F.normalize(ma, dim=1).numpy() if normalize else ma.numpy()
        b_arr = F.normalize(mb, dim=1).numpy() if normalize else mb.numpy()
        if n_max > 0 and len(a_arr) > n_max:
            rng = np.random.RandomState(hash(pk + 'a') & 0xFFFF)
            idx = rng.choice(len(a_arr), n_max, replace=False)
            a_arr = a_arr[idx]
            b_arr = b_arr[idx]
        info.append((sa, pk, len(a_arr)))
        info.append((sb, pk, len(b_arr)))
        blocks.append(a_arr)
        blocks.append(b_arr)

    combined = np.concatenate(blocks, axis=0)
    print(f"  [{split_name}] t-SNE on {len(combined)} points (perplexity={perplexity})")
    metric = 'cosine' if normalize else 'euclidean'
    coords = _make_tsne(combined, perplexity=perplexity, metric=metric)

    off = 0
    block_coords = []
    for (_, _, n) in info:
        block_coords.append(coords[off:off + n])
        off += n

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: by modality
    ax = axes[0]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=sensor_color[sensor],
                   alpha=0.4, s=12, linewidths=0, rasterized=True)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=s, markersize=8)
               for s, c in sensor_color.items()]
    ax.legend(handles=handles, fontsize=10, frameon=False)
    ax.set_title(f't-SNE [{split_name}] — by modality')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Right: pair links
    ax = axes[1]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=sensor_color[sensor],
                   alpha=0.3, s=10, linewidths=0, rasterized=True)
    rng = np.random.RandomState(0)
    for i, pk in enumerate(_PAIRS):
        pa = block_coords[2 * i]
        pb = block_coords[2 * i + 1]
        n_links = min(len(pa), len(pb), 500)
        idx = rng.choice(min(len(pa), len(pb)), n_links, replace=False)
        for k in idx:
            ax.plot([pa[k, 0], pb[k, 0]], [pa[k, 1], pb[k, 1]],
                    c=link_color[pk], alpha=0.15, lw=0.5)
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=f'{pk} link')
               for pk, c in link_color.items()]
    ax.legend(handles=handles, fontsize=10, frameon=False)
    ax.set_title(f't-SNE [{split_name}] — pair links')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_bytask(mu_pairs, task_labels_per_pair, split_name, out_path,
                     n_max=2000, normalize=True, perplexity=None):
    """t-SNE colored by task (plug/pnp/wipe), marker by sensor."""
    from matplotlib.lines import Line2D

    blocks = []
    task_per_point = []
    sensor_per_point = []

    for pk in _PAIRS:
        ma, mb = mu_pairs[pk]
        sa, sb = _PAIR_SENSORS[pk]
        labels_a, labels_b = task_labels_per_pair[pk]

        a_arr = F.normalize(ma, dim=1).numpy() if normalize else ma.numpy()
        b_arr = F.normalize(mb, dim=1).numpy() if normalize else mb.numpy()
        la = np.array(labels_a)
        lb = np.array(labels_b)

        if n_max > 0 and len(a_arr) > n_max:
            rng = np.random.RandomState(hash(pk + 'task') & 0xFFFF)
            idx = rng.choice(len(a_arr), n_max, replace=False)
            a_arr = a_arr[idx]
            b_arr = b_arr[idx]
            la = la[idx]
            lb = lb[idx]

        blocks.append(a_arr)
        blocks.append(b_arr)
        task_per_point.extend(la.tolist())
        task_per_point.extend(lb.tolist())
        sensor_per_point.extend([sa] * len(a_arr))
        sensor_per_point.extend([sb] * len(b_arr))

    combined = np.concatenate(blocks, axis=0)
    n_total = len(combined)
    if perplexity is None:
        perplexity = min(50, max(5, n_total // 6))
    print(f"  [{split_name}_bytask] t-SNE on {n_total} points (perplexity={perplexity})")

    metric = 'cosine' if normalize else 'euclidean'
    coords = _make_tsne(combined, perplexity=perplexity, metric=metric)

    task_per_point = np.array(task_per_point)
    sensor_per_point = np.array(sensor_per_point)
    unique_tasks = sorted(set(task_per_point))

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
                       alpha=0.5, s=20, linewidths=0, rasterized=True)

    ax.set_title(f't-SNE [{split_name}] — by task', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = []
    for t, c in task_cmap.items():
        handles.append(Line2D([], [], marker='o', ls='', color=c, markersize=8, label=t))
    for s, m in sensor_marker.items():
        handles.append(Line2D([], [], marker=m, ls='', color='gray', markersize=8, label=s))
    ax.legend(handles=handles, fontsize=9, frameon=False, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Reconstruction visualizations (on dynamic val data)
# ---------------------------------------------------------------------------

@torch.no_grad()
def viz_recon_daimon(val_de_dataset, model, device, out_path, n=8):
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
    plt.suptitle('Daimon recon (dynamic val, de_pair)')
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
            axes[i, j].bar(range(len(arr)), arr, color='steelblue', alpha=0.8)
            if i == 0:
                axes[i, j].set_title(ttl, fontsize=8)
            axes[i, j].tick_params(labelsize=6)
    plt.suptitle('eFlesh recon (dynamic val, de_pair)')
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
    plt.suptitle('FlexiTac recon (dynamic val, ef_pair)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Eval cross_modal_vae3 trained on dynamic paired data.')
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset_name', default='dataset_sachin_dynamic_paired',
                        help='Dataset entry from datasets_config_sachin.yaml')
    parser.add_argument('--objects', nargs='+', default=['plug', 'pnp', 'wipe'])
    parser.add_argument('--out_dir', default=None,
                        help='Default: <checkpoint_dir>/eval_dynamic/')
    parser.add_argument('--max_tsne_points', type=int, default=0,
                        help='Cap per-modality points for t-SNE. 0 = all.')
    parser.add_argument('--tsne_perplexity', type=int, default=None)
    parser.add_argument('--no_l2_norm', action='store_true')
    parser.add_argument('--ef_stride', type=int, default=1,
                        help='Match training ef_stride for correct split.')
    parser.add_argument('--de_stride', type=int, default=1)
    parser.add_argument('--fd_stride', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'eval_dynamic')
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

    # ── Load dynamic paired dataset (train ep0000-03, val ep0004) ──
    _pair_strides = {k: v for k, v in
                     (('de', args.de_stride), ('ef', args.ef_stride),
                      ('fd', args.fd_stride)) if v > 1}
    if _pair_strides:
        print(f"Per-pair frame strides: {_pair_strides}")

    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device='cpu', objects=args.objects,
        pair_strides=_pair_strides or None,
    )
    print('Train sizes:', {k: len(v) for k, v in paired_train.items()})
    print('Val sizes:  ', {k: len(v) for k, v in paired_val.items()})

    # ── Extract task labels per pair for bytask plots ──
    task_labels_train = {}
    task_labels_val = {}
    for pk in _PAIRS:
        labels_tr = _get_task_labels(paired_train[pk], pk)
        labels_val = _get_task_labels(paired_val[pk], pk)
        task_labels_train[pk] = (labels_tr, labels_tr)
        task_labels_val[pk] = (labels_val, labels_val)

    # ── Extract μ for train and val ──
    print('\nExtracting train μ (ep0000-03)...')
    mu_pairs_tr = {}
    for pk in _PAIRS:
        mu_pairs_tr[pk] = extract_pair_mu(paired_train[pk], pk, model, args.device)

    print('\nExtracting val μ (ep0004)...')
    mu_pairs_v = {}
    for pk in _PAIRS:
        mu_pairs_v[pk] = extract_pair_mu(paired_val[pk], pk, model, args.device)

    # ── Cosine similarity per pair (val) ──
    print('\nComputing cosine distributions (val)...')
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

    # ── Cross-modal retrieval on val ──
    print('\nComputing retrieval Top-k (val)...')
    retr_lines = ['Cross-modal retrieval (dynamic val ep0004, Top-1/5/10):']
    for pk in _PAIRS:
        ma, mb = mu_pairs_v[pk]
        ab = topk_retrieval(ma, mb)
        ba = topk_retrieval(mb, ma)
        sa, sb = _PAIR_SENSORS[pk]
        retr_lines.append(f"  {pk} {sa}→{sb}: " + '  '.join(f"top{k}={v:.4f}" for k, v in ab.items()))
        retr_lines.append(f"  {pk} {sb}→{sa}: " + '  '.join(f"top{k}={v:.4f}" for k, v in ba.items()))
    retr_txt = '\n'.join(retr_lines)
    print(retr_txt)
    with open(os.path.join(out_dir, 'retrieval.txt'), 'w') as f:
        f.write(retr_txt + '\n')

    # ── t-SNE plots ──
    normalize = not args.no_l2_norm
    n_max = args.max_tsne_points
    perp = args.tsne_perplexity

    print('\nComputing t-SNE (train, ep0000-03)...')
    plot_tsne_modality(mu_pairs_tr, 'train (ep0-3)', os.path.join(out_dir, 'tsne_train.png'),
                       n_max=n_max, normalize=normalize, perplexity=perp)

    print('\nComputing t-SNE (val, ep0004)...')
    plot_tsne_modality(mu_pairs_v, 'val (ep4)', os.path.join(out_dir, 'tsne_val.png'),
                       n_max=n_max, normalize=normalize, perplexity=perp)

    print('\nComputing t-SNE by task (train)...')
    plot_tsne_bytask(mu_pairs_tr, task_labels_train, 'train',
                     os.path.join(out_dir, 'tsne_train_bytask.png'),
                     n_max=n_max, normalize=normalize, perplexity=perp)

    print('\nComputing t-SNE by task (val)...')
    plot_tsne_bytask(mu_pairs_v, task_labels_val, 'val',
                     os.path.join(out_dir, 'tsne_val_bytask.png'),
                     n_max=n_max, normalize=normalize, perplexity=perp)

    # ── Reconstruction ──
    print('\nReconstruction visualizations (dynamic val)...')
    viz_recon_daimon(paired_val['de'], model, args.device,
                     os.path.join(out_dir, 'recon_daimon.png'))
    viz_recon_eflesh(paired_val['de'], model, args.device,
                     os.path.join(out_dir, 'recon_eflesh.png'))
    viz_recon_flexi(paired_val['ef'], model, args.device,
                    os.path.join(out_dir, 'recon_flexi.png'))

    print(f"\nDone. All outputs under: {out_dir}/")


if __name__ == '__main__':
    main()
