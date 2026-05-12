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
    python scripts/eval_cross_modal_vae3_sachin.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_<run_name>_<TS> \
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
from joint_embedding_learning.data.datasets_sachin import get_paired_tri_datasets_sachin
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

def _make_tsne(points, perplexity=30, seed=42):
    kw = dict(n_components=2, perplexity=perplexity, random_state=seed)
    if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) >= (1, 0):
        kw['max_iter'] = 1000
    else:
        kw['n_iter'] = 1000
    return TSNE(**kw).fit_transform(points)


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


def plot_tsne_dynamic(dyn_mus, out_path, n_max=2000):
    """t-SNE of dynamic-only μ across the three sensors — THE key plot.

    Success = the three clouds overlap; failure = three separate clusters.
    """
    blocks = []
    sensor_color = {'D': 'tab:blue', 'E': 'tab:orange', 'F': 'tab:green'}
    info = []
    for s in _SENSORS:
        arr = F.normalize(dyn_mus[s], dim=1).numpy()
        arr, _ = _subsample(arr, n_max, seed=hash('dyn_' + s) & 0xFFFF)
        blocks.append(arr)
        info.append((s, len(arr)))

    combined = np.concatenate(blocks, axis=0)
    print(f"  [dynamic] t-SNE on {len(combined)} points")
    coords = _make_tsne(combined)

    fig, ax = plt.subplots(figsize=(8, 7))
    off = 0
    for (s, n) in info:
        pts = coords[off:off + n]
        off += n
        ax.scatter(pts[:, 0], pts[:, 1], c=sensor_color[s], alpha=0.5, s=14, label=s)
    ax.set_title('t-SNE dynamic μ — should overlap across sensors (deployment regime)')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
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
        st, _ = _subsample(F.normalize(static_mu_per_sensor[s], dim=1).numpy(),
                           n_max, seed=hash(f'st_{s}') & 0xFFFF)
        dy, _ = _subsample(F.normalize(dyn_mus[s], dim=1).numpy(),
                           n_max, seed=hash(f'dy_{s}') & 0xFFFF)
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
    parser.add_argument('--objects', nargs='+', default=['cylinder'])
    parser.add_argument('--out_dir', default=None,
                        help='Default: <checkpoint_dir>/eval_sachin/')
    parser.add_argument('--max_tsne_points', type=int, default=2000)
    parser.add_argument('--max_mmd_points', type=int, default=2000,
                        help='Sample size per side for MMD numeric metric.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'eval_sachin')
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

    # ── Load paired (sachin) + dynamic datasets ──
    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device='cpu', objects=args.objects,
    )
    dyn_train, _dv, _ = get_all_contrastive_datasets(
        args.dynamic_dataset_name, device='cpu',
    )
    print('Paired val sizes:', {k: len(v) for k, v in paired_val.items()})
    print('Dynamic sizes:', {k: len(v) for k, v in dyn_train.items()})

    # ── Extract paired μ (train + val) ──
    print('Extracting paired-static μ...')
    mu_pairs_tr, mu_pairs_v = {}, {}
    for pk in _PAIRS:
        mu_pairs_tr[pk] = extract_pair_mu(paired_train[pk], pk, model, args.device)
        mu_pairs_v[pk] = extract_pair_mu(paired_val[pk], pk, model, args.device)

    # ── Extract dynamic μ (per modality, unpaired) ──
    print('Extracting dynamic μ...')
    dyn_mus = {s: extract_self_mu(dyn_train[s], s, model, args.device) for s in _SENSORS}

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
                      n_max=args.max_tsne_points)
    plot_tsne_static_vs_dynamic(
        static_mu_per_sensor, dyn_mus,
        os.path.join(out_dir, 'tsne_static_vs_dynamic.png'),
        n_max=min(args.max_tsne_points, 1500),
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
