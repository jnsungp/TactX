"""
Contact prediction eval for cross_modal_vae3 — WITHIN-sensor vs ACROSS-sensor
generalization (CTTP-style, arXiv:2410.11834).

Trains a frozen-μ linear probe on ONE sensor's TRAIN-split latents and evaluates
it on every sensor's VAL-split latents. The diagonal of the resulting
sensor×sensor matrix is "generalization within sensor"; the off-diagonal is
"generalization across sensors". The val split holds out whole episodes, so the
contact points seen at test time were never seen during representation learning
nor probe training.

The probe is a SINGLE linear layer trained with Adam (CTTP-style):
    μ → nn.Linear(latent_dim, out_dim) → logits/score
  - binary contact : out_dim=2, nn.CrossEntropyLoss
  - contact score  : out_dim=1, nn.MSELoss   (continuous norm ∈ [0,1])

Two targets, both per-sensor (each sensor predicts ITS OWN contact state):
  1. Contact (binary)  — linear + CrossEntropy -> accuracy / balanced accuracy
  2. Contact score     — linear + MSE          -> R² / MAE   (norm_a/norm_b ∈ [0,1])

Labels come from build_contact_labels.py (.pt files with per-sensor
contact_a/contact_b and norm_a/norm_b, keyed by (object, episode, frame)).

Sensors:  D=daimon  E=eflesh  F=flexi
Pairs:    de=(D,E)  ef=(E,F)  fd=(F,D)   (sensor D is pooled from de.a + fd.b, etc.)

Usage:
    python scripts/eval_cross_modal_vae3_contact.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_new_run_uniforce_20260523_041336 \
        --dataset_name dataset_new \
        --contact_label_root contact_labels/dataset_new \
        --device cuda:0
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, r2_score, mean_absolute_error,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from joint_embedding_learning.data.datasets_sachin import get_paired_tri_datasets_sachin
from joint_embedding_learning.utils.models_setup import get_model

_SENSORS = ('D', 'E', 'F')
_PAIRS = ('de', 'ef', 'fd')
_PAIR_SENSORS = {'de': ('D', 'E'), 'ef': ('E', 'F'), 'fd': ('F', 'D')}
_PAIR_DIR = {'de': 'de_pair', 'ef': 'ef_pair', 'fd': 'fd_pair'}
_DEFAULT_OBJECTS = ['circle', 'cylinder', 'cylinder_vertical', 'ellipse',
                    'large_ball', 'plane', 'plane_horizontal', 'small_ball',
                    'square', 'triangle']


# ---------------------------------------------------------------------------
# μ extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_pair_mu(dataset, pair_name, model, device, batch_size=128):
    """Return (mu_a, mu_b, obj_ids, ep_ids, fr_ids) for one pair."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    mu_a_all, mu_b_all, obj_all, ep_all, fr_all = [], [], [], [], []
    for (xa, xb), labels in tqdm(loader, desc=f"paired_{pair_name}", leave=False):
        out = model.forward_pair(xa.to(device), xb.to(device), pair_name)
        mu_a_all.append(out['mu_a'].cpu())
        mu_b_all.append(out['mu_b'].cpu())
        obj_id, ep_id, fr_id = labels
        obj_all.append(obj_id.cpu())
        ep_all.append(ep_id.cpu())
        fr_all.append(fr_id.cpu())
    return (torch.cat(mu_a_all), torch.cat(mu_b_all),
            torch.cat(obj_all), torch.cat(ep_all), torch.cat(fr_all))


def _unwrap(d):
    while isinstance(d, Subset):
        d = d.dataset
    return d


# ---------------------------------------------------------------------------
# Label loading + per-sensor aggregation
# ---------------------------------------------------------------------------

def _load_pair_labels(label_root, pair_name, object_names):
    """{(obj_id, ep, fr): (contact_a, contact_b, norm_a, norm_b)} for one pair.

    obj_id indexes into `object_names` (the per-pair index→name list), which is
    the same convention the dataset uses, so meta obj_ids line up directly."""
    pair_dir = _PAIR_DIR[pair_name]
    label_map = {}
    for obj_id, obj in enumerate(object_names):
        ep_dir = os.path.join(label_root, pair_dir, obj)
        if not os.path.isdir(ep_dir):
            continue
        for f in sorted(glob.glob(os.path.join(ep_dir, 'ep*.pt'))):
            d = torch.load(f, map_location='cpu', weights_only=False)
            ep = int(d['episode'])
            frames = np.asarray(d['frames'])
            ca = np.asarray(d['contact_a']).astype(np.int64)
            cb = np.asarray(d['contact_b']).astype(np.int64)
            na = np.asarray(d['norm_a']).astype(np.float32)
            nb = np.asarray(d['norm_b']).astype(np.float32)
            for i, fr in enumerate(frames):
                label_map[(obj_id, ep, int(fr))] = (
                    int(ca[i]), int(cb[i]), float(na[i]), float(nb[i]))
    return label_map


def build_per_sensor(mu_pairs, meta, object_names_per_pair, label_root):
    """Aggregate {sensor: {'X', 'contact', 'score'}} keeping only frames that
    have a contact label. Each sensor predicts its OWN contact/score; a sensor
    is pooled from both pairs it participates in (e.g. D = de.a ⊕ fd.b)."""
    bucket = {s: {'X': [], 'contact': [], 'score': []} for s in _SENSORS}
    missing_pairs, skipped = [], 0
    for pk in _PAIRS:
        labels = _load_pair_labels(label_root, pk, object_names_per_pair[pk])
        if not labels:
            missing_pairs.append(pk)
            continue
        ma, mb = mu_pairs[pk]
        obj_np, ep_np, fr_np = (t.numpy() for t in meta[pk])
        sa, sb = _PAIR_SENSORS[pk]
        n = ma.shape[0]
        ca = np.zeros(n, np.int64); cb = np.zeros(n, np.int64)
        na = np.zeros(n, np.float32); nb = np.zeros(n, np.float32)
        valid = np.zeros(n, bool)
        for i in range(n):
            key = (int(obj_np[i]), int(ep_np[i]), int(fr_np[i]))
            if key in labels:
                ca[i], cb[i], na[i], nb[i] = labels[key]
                valid[i] = True
            else:
                skipped += 1
        if not valid.any():
            continue
        bucket[sa]['X'].append(ma[valid].numpy())
        bucket[sa]['contact'].append(ca[valid]); bucket[sa]['score'].append(na[valid])
        bucket[sb]['X'].append(mb[valid].numpy())
        bucket[sb]['contact'].append(cb[valid]); bucket[sb]['score'].append(nb[valid])

    out = {}
    for s in _SENSORS:
        if bucket[s]['X']:
            out[s] = {
                'X': np.concatenate(bucket[s]['X'], axis=0),
                'contact': np.concatenate(bucket[s]['contact'], axis=0),
                'score': np.concatenate(bucket[s]['score'], axis=0),
            }
    return out, missing_pairs, skipped


def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Single-layer linear probe (CTTP-style): μ → nn.Linear → logits/score
# ---------------------------------------------------------------------------

class LinearProbe:
    """One nn.Linear trained with Adam. task='cls' -> CrossEntropyLoss,
    task='reg' -> MSELoss. Predicts on arbitrary X so one fit can be evaluated
    on every test sensor."""

    def __init__(self, in_dim, out_dim, task, device,
                 epochs=150, lr=1e-3, weight_decay=0.0, batch_size=4096, seed=0):
        self.in_dim, self.out_dim, self.task = in_dim, out_dim, task
        self.device = device
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.batch_size, self.seed = batch_size, seed
        self.layer = None

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.layer = nn.Linear(self.in_dim, self.out_dim).to(self.device)
        opt = torch.optim.Adam(self.layer.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        Xt = torch.from_numpy(np.ascontiguousarray(X)).float().to(self.device)
        if self.task == 'cls':
            yt = torch.from_numpy(y.astype(np.int64)).to(self.device)
            crit = nn.CrossEntropyLoss()
        else:
            yt = torch.from_numpy(y.astype(np.float32)).to(self.device).unsqueeze(1)
            crit = nn.MSELoss()
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
    def predict(self, X):
        self.layer.eval()
        Xt = torch.from_numpy(np.ascontiguousarray(X)).float().to(self.device)
        out = self.layer(Xt)
        if self.task == 'cls':
            return out.argmax(dim=1).cpu().numpy()
        return out.squeeze(1).cpu().numpy()


# ---------------------------------------------------------------------------
# Generalization matrices (train sensor → test sensor)
# ---------------------------------------------------------------------------

def binary_matrix(data_tr, data_v, normalize, device, cfg):
    """3×3 dicts {(train,test): value} for accuracy & balanced accuracy."""
    acc, bal = {}, {}
    for s_tr in _SENSORS:
        if s_tr not in data_tr:
            continue
        Xtr = data_tr[s_tr]['X']; ytr = data_tr[s_tr]['contact']
        if len(np.unique(ytr)) < 2:
            continue
        if normalize:
            Xtr = _l2(Xtr)
        probe = LinearProbe(Xtr.shape[1], 2, 'cls', device, **cfg).fit(Xtr, ytr)
        for s_te in _SENSORS:
            if s_te not in data_v:
                continue
            Xte = data_v[s_te]['X']; yte = data_v[s_te]['contact']
            if normalize:
                Xte = _l2(Xte)
            yp = probe.predict(Xte)
            acc[(s_tr, s_te)] = accuracy_score(yte, yp)
            bal[(s_tr, s_te)] = balanced_accuracy_score(yte, yp)
    return acc, bal


def score_matrix(data_tr, data_v, normalize, device, cfg):
    """3×3 dicts {(train,test): value} for R² & MAE of contact-score regression."""
    r2, mae = {}, {}
    for s_tr in _SENSORS:
        if s_tr not in data_tr:
            continue
        Xtr = data_tr[s_tr]['X']; ytr = data_tr[s_tr]['score']
        if normalize:
            Xtr = _l2(Xtr)
        probe = LinearProbe(Xtr.shape[1], 1, 'reg', device, **cfg).fit(Xtr, ytr)
        for s_te in _SENSORS:
            if s_te not in data_v:
                continue
            Xte = data_v[s_te]['X']; yte = data_v[s_te]['score']
            if normalize:
                Xte = _l2(Xte)
            yp = probe.predict(Xte)
            r2[(s_tr, s_te)] = float(r2_score(yte, yp))
            mae[(s_tr, s_te)] = float(mean_absolute_error(yte, yp))
    return r2, mae


# ---------------------------------------------------------------------------
# Formatting + plotting
# ---------------------------------------------------------------------------

def fmt_matrix(mat, fmt='{:.3f}'):
    lines = ['  train\\test ' + ''.join(f'{s:>9}' for s in _SENSORS)]
    for s_tr in _SENSORS:
        cells = []
        for s_te in _SENSORS:
            cells.append(fmt.format(mat[(s_tr, s_te)]) if (s_tr, s_te) in mat else 'n/a')
        lines.append(f'  {s_tr:>9} ' + ''.join(f'{c:>9}' for c in cells))
    return '\n'.join(lines)


def summarize(mat):
    diag = [mat[(s, s)] for s in _SENSORS if (s, s) in mat]
    off = [mat[(a, b)] for a in _SENSORS for b in _SENSORS
           if a != b and (a, b) in mat]
    within = float(np.mean(diag)) if diag else float('nan')
    across = float(np.mean(off)) if off else float('nan')
    return within, across


def _to_grid(mat):
    M = np.full((len(_SENSORS), len(_SENSORS)), np.nan)
    for i, a in enumerate(_SENSORS):
        for j, b in enumerate(_SENSORS):
            if (a, b) in mat:
                M[i, j] = mat[(a, b)]
    return M


def plot_panel(specs, suptitle, out_path):
    """specs: list of (title, mat, fmt, vmin, vmax, cmap)."""
    fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 4))
    if len(specs) == 1:
        axes = [axes]
    for ax, (title, mat, fmt, vmin, vmax, cmap) in zip(axes, specs):
        M = _to_grid(mat)
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        ax.set_xticks(range(len(_SENSORS))); ax.set_xticklabels(_SENSORS)
        ax.set_yticks(range(len(_SENSORS))); ax.set_yticklabels(_SENSORS)
        ax.set_xlabel('test sensor'); ax.set_ylabel('train sensor')
        ax.set_title(title, fontsize=11)
        lo = vmin if vmin is not None else np.nanmin(M)
        hi = vmax if vmax is not None else np.nanmax(M)
        mid = (lo + hi) / 2
        for i in range(len(_SENSORS)):
            for j in range(len(_SENSORS)):
                if not np.isnan(M[i, j]):
                    color = 'white' if M[i, j] < mid else 'black'
                    ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center',
                            color=color, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(suptitle, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Contact within/across-sensor eval')
    ap.add_argument('--checkpoint_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dataset_name', default='dataset_new')
    ap.add_argument('--objects', nargs='+', default=_DEFAULT_OBJECTS)
    ap.add_argument('--contact_label_root', default='contact_labels/dataset_new')
    ap.add_argument('--probe_epochs', type=int, default=150)
    ap.add_argument('--probe_lr', type=float, default=1e-3)
    ap.add_argument('--probe_wd', type=float, default=0.0)
    ap.add_argument('--probe_batch_size', type=int, default=4096)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    probe_cfg = dict(epochs=args.probe_epochs, lr=args.probe_lr,
                     weight_decay=args.probe_wd,
                     batch_size=args.probe_batch_size, seed=args.seed)

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'eval_contact')
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── model ──
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

    # ── datasets (same train/val episode split as training) ──
    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device='cpu', objects=args.objects)
    print('Train sizes:', {k: len(v) for k, v in paired_train.items()})
    print('Val sizes:  ', {k: len(v) for k, v in paired_val.items()})

    # ── extract μ + meta ──
    print('\n=== Extracting μ ===')
    mu_tr, mu_v, meta_tr, meta_v = {}, {}, {}, {}
    for pk in _PAIRS:
        ma, mb, o, e, fr = extract_pair_mu(paired_train[pk], pk, model, args.device)
        mu_tr[pk] = (ma, mb); meta_tr[pk] = (o, e, fr)
        ma, mb, o, e, fr = extract_pair_mu(paired_val[pk], pk, model, args.device)
        mu_v[pk] = (ma, mb); meta_v[pk] = (o, e, fr)
    object_names_per_pair = {
        pk: list(_unwrap(paired_train[pk]).object_names) for pk in _PAIRS}

    # ── per-sensor data ──
    data_tr, miss_tr, skip_tr = build_per_sensor(
        mu_tr, meta_tr, object_names_per_pair, args.contact_label_root)
    data_v, miss_v, skip_v = build_per_sensor(
        mu_v, meta_v, object_names_per_pair, args.contact_label_root)
    if not data_tr or not data_v:
        raise SystemExit(
            f"No contact labels found under {args.contact_label_root} "
            f"(missing pairs tr={miss_tr}, v={miss_v}).")

    counts = ['Per-sensor frame counts (with contact label):']
    for s in _SENSORS:
        ntr = len(data_tr[s]['contact']) if s in data_tr else 0
        nv = len(data_v[s]['contact']) if s in data_v else 0
        ptr = float(data_tr[s]['contact'].mean()) if s in data_tr else float('nan')
        pv = float(data_v[s]['contact'].mean()) if s in data_v else float('nan')
        counts.append(f"  {s}: n_train={ntr:7d} (pos={ptr:.3f})  "
                      f"n_val={nv:7d} (pos={pv:.3f})")
    counts.append(f"  skipped (no label): train={skip_tr}  val={skip_v}")
    counts_txt = '\n'.join(counts)
    print(counts_txt)

    # ── build matrices (raw + L2-normalized μ) ──
    sections = [counts_txt]
    for normalize, tag in [(False, 'raw μ'), (True, 'L2-normalized μ')]:
        b_acc, b_bal = binary_matrix(data_tr, data_v, normalize,
                                     args.device, probe_cfg)
        s_r2, s_mae = score_matrix(data_tr, data_v, normalize,
                                   args.device, probe_cfg)

        wa_acc, ac_acc = summarize(b_acc)
        wa_bal, ac_bal = summarize(b_bal)
        wa_r2, ac_r2 = summarize(s_r2)
        wa_mae, ac_mae = summarize(s_mae)

        block = [
            f"\n{'=' * 70}",
            f"[{tag}]  (rows = TRAIN sensor, cols = TEST sensor)",
            f"{'=' * 70}",
            "\n-- Contact BINARY: accuracy --",
            fmt_matrix(b_acc),
            f"  within-sensor (diag) mean = {wa_acc:.4f}   "
            f"across-sensor (off-diag) mean = {ac_acc:.4f}",
            "\n-- Contact BINARY: balanced accuracy --",
            fmt_matrix(b_bal),
            f"  within-sensor mean = {wa_bal:.4f}   across-sensor mean = {ac_bal:.4f}",
            "\n-- Contact SCORE: R² --",
            fmt_matrix(s_r2),
            f"  within-sensor mean = {wa_r2:.4f}   across-sensor mean = {ac_r2:.4f}",
            "\n-- Contact SCORE: MAE --",
            fmt_matrix(s_mae, fmt='{:.4f}'),
            f"  within-sensor mean = {wa_mae:.4f}   across-sensor mean = {ac_mae:.4f}",
        ]
        sections.append('\n'.join(block))

        suffix = 'norm' if normalize else 'raw'
        plot_panel(
            [('contact acc', b_acc, '{:.2f}', 0.0, 1.0, 'viridis'),
             ('contact bal-acc', b_bal, '{:.2f}', 0.0, 1.0, 'viridis'),
             ('score R²', s_r2, '{:.2f}', -1.0, 1.0, 'coolwarm_r'),
             ('score MAE', s_mae, '{:.3f}', None, None, 'magma_r')],
            f'Contact within/across sensor — {tag}',
            os.path.join(out_dir, f'contact_matrices_{suffix}.png'))

    metrics_txt = '\n'.join(sections)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(metrics_txt + '\n')
    print(f"\nWrote {os.path.join(out_dir, 'metrics.txt')}")
    print(f"Done. Outputs under: {out_dir}/")


if __name__ == '__main__':
    main()
