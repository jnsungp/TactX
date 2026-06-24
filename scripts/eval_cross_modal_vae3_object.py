"""
Object-classification eval for cross_modal_vae3 — WITHIN-sensor vs ACROSS-sensor
generalization (CTTP-style, arXiv:2410.11834).

Trains a frozen-μ linear classifier on ONE sensor's TRAIN-split latents and
evaluates it on every sensor's VAL-split latents (10 objects). The diagonal of
the sensor×sensor accuracy matrix is "generalization within sensor"; the
off-diagonal is "generalization across sensors". The val split holds out whole
episodes, so the grasps / contact points classified at test time were never seen
during representation learning nor classifier training (mirrors the paper's
"unseen grasps" protocol).

The classifier is a SINGLE linear layer trained with Adam (CTTP-style):
    μ → nn.Linear(latent_dim, n_objects) → logits → nn.CrossEntropyLoss

Also saves a t-SNE of the VAL latents colored by the 10 objects (sensors drawn
with distinct markers) — the analogue of Fig. 4 in the paper.

Sensors:  D=daimon  E=eflesh  F=flexi
Pairs:    de=(D,E)  ef=(E,F)  fd=(F,D)   (sensor D is pooled from de.a + fd.b, etc.)

Usage:
    python scripts/eval_cross_modal_vae3_object.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/0_TACTX_nb \
        --dataset_name dataset_new_concat \
        --device cuda:0
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
import sklearn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from joint_embedding_learning.data.datasets_sachin import get_paired_tri_datasets_sachin
from joint_embedding_learning.utils.models_setup import get_model

_SENSORS = ('D', 'E', 'F')
_PAIRS = ('de', 'ef', 'fd')
_PAIR_SENSORS = {'de': ('D', 'E'), 'ef': ('E', 'F'), 'fd': ('F', 'D')}
_DEFAULT_OBJECTS = ['circle', 'cylinder', 'cylinder_vertical', 'ellipse',
                    'large_ball', 'plane', 'plane_horizontal', 'small_ball',
                    'square', 'triangle']


# ---------------------------------------------------------------------------
# μ extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_pair_mu(dataset, pair_name, model, device, batch_size=128):
    """Return (mu_a, mu_b, obj_ids) for one pair."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    mu_a_all, mu_b_all, obj_all = [], [], []
    for (xa, xb), labels in tqdm(loader, desc=f"paired_{pair_name}", leave=False):
        out = model.forward_pair(xa.to(device), xb.to(device), pair_name)
        mu_a_all.append(out['mu_a'].cpu())
        mu_b_all.append(out['mu_b'].cpu())
        obj_all.append(labels[0].cpu())
    return torch.cat(mu_a_all), torch.cat(mu_b_all), torch.cat(obj_all)


def _unwrap(d):
    while isinstance(d, Subset):
        d = d.dataset
    return d


def build_per_sensor(mu_pairs, meta_obj, object_names_per_pair, canonical):
    """Aggregate {sensor: {'X', 'obj'}} where obj is a CANONICAL object id
    (index into `canonical`). A sensor is pooled from both pairs it appears in
    (e.g. D = de.a ⊕ fd.b). Per-pair obj ids are remapped via each pair's
    own name list, so pooling is robust even if id orderings ever differ."""
    bucket = {s: {'X': [], 'obj': []} for s in _SENSORS}
    canon_id = {name: i for i, name in enumerate(canonical)}
    for pk in _PAIRS:
        names = object_names_per_pair[pk]
        remap = np.array([canon_id[n] for n in names], dtype=np.int64)
        ma, mb = mu_pairs[pk]
        gobj = remap[meta_obj[pk].numpy()]
        sa, sb = _PAIR_SENSORS[pk]
        bucket[sa]['X'].append(ma.numpy()); bucket[sa]['obj'].append(gobj.copy())
        bucket[sb]['X'].append(mb.numpy()); bucket[sb]['obj'].append(gobj.copy())
    out = {}
    for s in _SENSORS:
        out[s] = {
            'X': np.concatenate(bucket[s]['X'], axis=0),
            'obj': np.concatenate(bucket[s]['obj'], axis=0),
        }
    return out


def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Single-layer linear probe (CTTP-style): μ → nn.Linear → logits
# ---------------------------------------------------------------------------

class LinearProbe:
    """One nn.Linear classifier trained with Adam + CrossEntropyLoss. Predicts
    on arbitrary X so one fit can be evaluated on every test sensor."""

    def __init__(self, in_dim, out_dim, device,
                 epochs=150, lr=1e-3, weight_decay=0.0, batch_size=4096, seed=0):
        self.in_dim, self.out_dim, self.device = in_dim, out_dim, device
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.batch_size, self.seed = batch_size, seed
        self.layer = None

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.layer = nn.Linear(self.in_dim, self.out_dim).to(self.device)
        opt = torch.optim.Adam(self.layer.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        crit = nn.CrossEntropyLoss()
        Xt = torch.from_numpy(np.ascontiguousarray(X)).float().to(self.device)
        yt = torch.from_numpy(y.astype(np.int64)).to(self.device)
        n, bs = Xt.shape[0], self.batch_size
        self.layer.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = crit(self.layer(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_logits(self, X):
        self.layer.eval()
        Xt = torch.from_numpy(np.ascontiguousarray(X)).float().to(self.device)
        return self.layer(Xt).cpu().numpy()


def _topk_acc(logits, y_true, k):
    """Fraction of samples whose true label is among the top-k predictions."""
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


# ---------------------------------------------------------------------------
# Generalization matrix (train sensor → test sensor)
# ---------------------------------------------------------------------------

def object_matrix(data_tr, data_v, normalize, num_classes, device, cfg,
                  topks=(1, 2, 3)):
    """Per-(train,test) sensor matrices for top-k accuracy (k in `topks`) plus
    balanced top-1 accuracy. Returns ({f'top{k}': mat}, bal_mat)."""
    mats = {f'top{k}': {} for k in topks}
    bal = {}
    for s_tr in _SENSORS:
        Xtr = data_tr[s_tr]['X']; ytr = data_tr[s_tr]['obj']
        if normalize:
            Xtr = _l2(Xtr)
        probe = LinearProbe(Xtr.shape[1], num_classes, device, **cfg).fit(Xtr, ytr)
        for s_te in _SENSORS:
            Xte = data_v[s_te]['X']; yte = data_v[s_te]['obj']
            if normalize:
                Xte = _l2(Xte)
            logits = probe.predict_logits(Xte)
            for k in topks:
                mats[f'top{k}'][(s_tr, s_te)] = _topk_acc(logits, yte, k)
            bal[(s_tr, s_te)] = balanced_accuracy_score(yte, logits.argmax(axis=1))
    return mats, bal


# ---------------------------------------------------------------------------
# Formatting + plotting
# ---------------------------------------------------------------------------

def fmt_matrix(mat, fmt='{:.3f}'):
    lines = ['  train\\test ' + ''.join(f'{s:>9}' for s in _SENSORS)]
    for s_tr in _SENSORS:
        cells = [fmt.format(mat[(s_tr, s_te)]) if (s_tr, s_te) in mat else 'n/a'
                 for s_te in _SENSORS]
        lines.append(f'  {s_tr:>9} ' + ''.join(f'{c:>9}' for c in cells))
    return '\n'.join(lines)


def summarize(mat):
    diag = [mat[(s, s)] for s in _SENSORS if (s, s) in mat]
    off = [mat[(a, b)] for a in _SENSORS for b in _SENSORS
           if a != b and (a, b) in mat]
    return (float(np.mean(diag)) if diag else float('nan'),
            float(np.mean(off)) if off else float('nan'))


def _to_grid(mat):
    M = np.full((len(_SENSORS), len(_SENSORS)), np.nan)
    for i, a in enumerate(_SENSORS):
        for j, b in enumerate(_SENSORS):
            if (a, b) in mat:
                M[i, j] = mat[(a, b)]
    return M


def plot_panel(specs, suptitle, out_path, chance=None):
    fig, axes = plt.subplots(1, len(specs), figsize=(4.4 * len(specs), 4))
    if len(specs) == 1:
        axes = [axes]
    for ax, (title, mat) in zip(axes, specs):
        M = _to_grid(mat)
        im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap='viridis', aspect='equal')
        ax.set_xticks(range(len(_SENSORS))); ax.set_xticklabels(_SENSORS)
        ax.set_yticks(range(len(_SENSORS))); ax.set_yticklabels(_SENSORS)
        ax.set_xlabel('test sensor'); ax.set_ylabel('train sensor')
        t = title + (f'\n(chance={chance:.3f})' if chance else '')
        ax.set_title(t, fontsize=11)
        for i in range(len(_SENSORS)):
            for j in range(len(_SENSORS)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, '{:.2f}'.format(M[i, j]), ha='center',
                            va='center',
                            color='white' if M[i, j] < 0.5 else 'black',
                            fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(suptitle, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def _make_tsne(points, perplexity, seed=42):
    kw = dict(n_components=2, perplexity=perplexity, init='pca',
              learning_rate='auto', metric='cosine', random_state=seed)
    if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) >= (1, 0):
        kw['max_iter'] = 2000
    else:
        kw['n_iter'] = 2000
    coords = TSNE(**kw).fit_transform(points)
    return coords - coords.mean(axis=0, keepdims=True)


def _auto_perp(n, user):
    """Fewer points -> larger relative perplexity for tighter clusters."""
    return user if user is not None else int(min(50, max(15, n // 40)))


def _prep_per_sensor(data, n_max, seed):
    """{sensor: (X_l2, obj)} with each sensor capped at n_max (random subset)."""
    rng = np.random.RandomState(seed)
    out = {}
    for s in _SENSORS:
        Xs = _l2(data[s]['X']); ys = data[s]['obj']
        if n_max > 0 and len(Xs) > n_max:
            idx = rng.choice(len(Xs), n_max, replace=False)
            Xs, ys = Xs[idx], ys[idx]
        out[s] = (Xs, ys)
    return out


def _scatter_by_object(ax, coords, obj, n_obj, obj_color, size, alpha):
    for oid in range(n_obj):
        m = obj == oid
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], c=[obj_color[oid]],
                       s=size, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_tsne_object(data_v, canonical, out_path, out_path_per_sensor,
                     n_max=1500, perplexity=None, seed=42,
                     point_size=45, alpha=0.35):
    """Object-colored t-SNE. The per-sensor figure runs an INDEPENDENT t-SNE per
    sensor, so each panel shows object clusters without the sensor axis dominating
    the layout (the combined pooled embedding is laid out mostly by sensor, which
    is why object colors look mixed there even when the linear probe scores well).
    """
    per = _prep_per_sensor(data_v, n_max, seed)
    n_obj = len(canonical)
    cmap = plt.get_cmap('tab10' if n_obj <= 10 else 'tab20')
    obj_color = {i: cmap(i % cmap.N) for i in range(n_obj)}
    obj_handles = [Line2D([], [], marker='o', ls='', color=obj_color[i],
                          markersize=11, label=canonical[i]) for i in range(n_obj)]

    # ── per-sensor INDEPENDENT embeddings (clustered by object) ──
    fig, axes = plt.subplots(1, len(_SENSORS), figsize=(7 * len(_SENSORS), 6.8))
    for ax, s in zip(axes, _SENSORS):
        Xs, ys = per[s]
        perp = _auto_perp(len(Xs), perplexity)
        print(f"  [sensor {s}] t-SNE on {len(Xs)} val points (perplexity={perp})")
        coords = _make_tsne(Xs, perp, seed)
        _scatter_by_object(ax, coords, ys, n_obj, obj_color, point_size, alpha)
        ax.set_title(f'sensor {s}  (n={len(Xs)})', fontsize=14)
    axes[-1].legend(handles=obj_handles, fontsize=10, loc='center left',
                    bbox_to_anchor=(1.02, 0.5), title='object')
    plt.suptitle('Per-sensor t-SNE of val latents (independent embeddings) — '
                 'colored by object', y=1.02, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path_per_sensor, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path_per_sensor}")

    # ── combined pooled embedding (sensor-agnostic view) ──
    X = np.concatenate([per[s][0] for s in _SENSORS], axis=0)
    obj = np.concatenate([per[s][1] for s in _SENSORS], axis=0)
    perp = _auto_perp(len(X), perplexity)
    print(f"  [combined] t-SNE on {len(X)} val points (perplexity={perp})")
    coords = _make_tsne(X, perp, seed)
    fig, ax = plt.subplots(figsize=(10, 9))
    _scatter_by_object(ax, coords, obj, n_obj, obj_color, point_size, alpha)
    ax.legend(handles=obj_handles, fontsize=10, loc='upper left', ncol=2,
              title='object', framealpha=0.9)
    ax.set_title('t-SNE of val latents (all sensors pooled) — colored by object',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Object within/across-sensor eval')
    ap.add_argument('--checkpoint_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dataset_name', default='dataset_new')
    ap.add_argument('--objects', nargs='+', default=_DEFAULT_OBJECTS)
    ap.add_argument('--probe_epochs', type=int, default=150)
    ap.add_argument('--probe_lr', type=float, default=1e-3)
    ap.add_argument('--probe_wd', type=float, default=0.0)
    ap.add_argument('--probe_batch_size', type=int, default=4096)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--max_tsne_points', type=int, default=1500,
                    help='Max val points per sensor for t-SNE (0 = all). Fewer '
                         'points give tighter, more readable clusters.')
    ap.add_argument('--tsne_perplexity', type=int, default=None)
    ap.add_argument('--tsne_point_size', type=float, default=45.0)
    ap.add_argument('--tsne_alpha', type=float, default=0.35)
    ap.add_argument('--contrastive_only', action='store_true',
                    help='Evaluate a CrossModalCLR3 (contrastive-only, no '
                         'decoders) checkpoint instead of CrossModalVAE3. '
                         'Object classification uses only μ, so no other '
                         'changes are needed.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    probe_cfg = dict(epochs=args.probe_epochs, lr=args.probe_lr,
                     weight_decay=args.probe_wd,
                     batch_size=args.probe_batch_size, seed=args.seed)

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'eval_object')
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── model ──
    model_name = 'cross_modal_clr3' if args.contrastive_only else 'cross_modal_vae3'
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model(model_name, model_details).to(args.device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'model.pt'),
                      map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded {args.checkpoint_dir}/model.pt  (model={model_name})")

    # ── datasets (same train/val episode split as training) ──
    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device='cpu', objects=args.objects)
    print('Train sizes:', {k: len(v) for k, v in paired_train.items()})
    print('Val sizes:  ', {k: len(v) for k, v in paired_val.items()})

    # ── extract μ + object ids ──
    print('\n=== Extracting μ ===')
    mu_tr, mu_v, obj_tr, obj_v = {}, {}, {}, {}
    for pk in _PAIRS:
        ma, mb, o = extract_pair_mu(paired_train[pk], pk, model, args.device)
        mu_tr[pk] = (ma, mb); obj_tr[pk] = o
        ma, mb, o = extract_pair_mu(paired_val[pk], pk, model, args.device)
        mu_v[pk] = (ma, mb); obj_v[pk] = o
    object_names_per_pair = {
        pk: list(_unwrap(paired_train[pk]).object_names) for pk in _PAIRS}
    canonical = sorted(args.objects)

    data_tr = build_per_sensor(mu_tr, obj_tr, object_names_per_pair, canonical)
    data_v = build_per_sensor(mu_v, obj_v, object_names_per_pair, canonical)

    counts = [f'Objects ({len(canonical)}): {canonical}',
              'Per-sensor frame counts:']
    for s in _SENSORS:
        counts.append(f"  {s}: n_train={len(data_tr[s]['obj']):7d}  "
                      f"n_val={len(data_v[s]['obj']):7d}")
    chance = 1.0 / len(canonical)
    counts.append(f"  random-chance top-1 accuracy = {chance:.3f}")
    counts_txt = '\n'.join(counts)
    print(counts_txt)

    # ── matrices (raw + L2-normalized μ) ──
    topks = (1, 2, 3)
    sections = [counts_txt]
    for normalize, tag in [(False, 'raw μ'), (True, 'L2-normalized μ')]:
        mats, bal = object_matrix(data_tr, data_v, normalize,
                                  len(canonical), args.device, probe_cfg,
                                  topks=topks)
        block = [
            f"\n{'=' * 70}",
            f"[{tag}]  object classification (rows = TRAIN sensor, cols = TEST sensor)",
            f"{'=' * 70}",
        ]
        for k in topks:
            wa, ac = summarize(mats[f'top{k}'])
            block += [
                f"\n-- top-{k} accuracy --",
                fmt_matrix(mats[f'top{k}']),
                f"  within-sensor (diag) mean = {wa:.4f}   "
                f"across-sensor (off-diag) mean = {ac:.4f}",
            ]
        wa_bal, ac_bal = summarize(bal)
        block += [
            "\n-- balanced accuracy (top-1) --",
            fmt_matrix(bal),
            f"  within-sensor mean = {wa_bal:.4f}   across-sensor mean = {ac_bal:.4f}",
        ]
        sections.append('\n'.join(block))
        suffix = 'norm' if normalize else 'raw'
        plot_panel([(f'top-{k} accuracy', mats[f'top{k}']) for k in topks]
                   + [('balanced acc (top-1)', bal)],
                   f'Object classification within/across sensor — {tag}',
                   os.path.join(out_dir, f'object_matrices_{suffix}.png'))

    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write('\n'.join(sections) + '\n')
    print(f"\nWrote {os.path.join(out_dir, 'metrics.txt')}")

    # ── t-SNE colored by object (val) ──
    print('\n=== t-SNE (val, colored by object) ===')
    plot_tsne_object(
        data_v, canonical,
        os.path.join(out_dir, 'tsne_object_val.png'),
        os.path.join(out_dir, 'tsne_object_val_per_sensor.png'),
        n_max=args.max_tsne_points, perplexity=args.tsne_perplexity,
        seed=args.seed + 42,
        point_size=args.tsne_point_size, alpha=args.tsne_alpha)

    print(f"Done. Outputs under: {out_dir}/")


if __name__ == '__main__':
    main()
