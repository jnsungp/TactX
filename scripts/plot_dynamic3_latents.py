"""
Latent analysis for the 3 paired dynamic datasets in
latent_data_processed_dynamic/ : plug, pnp, wipe.

Produces three figures:
  A. <out>_combined_tsne.png   — all 3 datasets pooled, one t-SNE.
        colour = dataset (plug/pnp/wipe), marker = sensor (D=o E=^ F=s)
  B. <out>_per_obj_sensor_tsne.png — 3x3 grid (rows = dataset, cols = D/E/F),
        each cell an individual t-SNE of that (dataset, sensor) latent cloud,
        coloured by episode.
  C. <out>_per_episode_cossim.png  — 3x3 grid (rows = dataset, cols = de/ef/fd),
        per-episode mean matched-pair cosine similarity (s0[i] vs s1[i]).

Model + z-score stats come from --checkpoint_dir (its saved
dataset_details.yaml — what the model trained with). Data is z-scored / rot180
exactly as in training. ef pair is frame-strided by --ef_stride.

Usage:
    python scripts/plot_dynamic3_latents.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_static12_run_infonce_static12_efs3_20260517_224612 \
        --weights model_ep400.pt --device cuda:0
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_sachin_pair_similarity import _load_model, _build_pair_dataset

_NAME2LETTER = {'daimon': 'D', 'eflesh': 'E', 'flexi': 'F'}
_DATASET_COLOR = {'plug': '#2171b5', 'pnp': '#e6261f', 'wipe': '#2ca02c'}
_SENSOR_MARKER = {'D': 'o', 'E': '^', 'F': 's'}


@torch.no_grad()
def encode_task(task, dd, dynamic_root, model, device, ef_stride, batch_size):
    """Encode one dynamic dataset. Returns pair_mu[pk] = (mu_a, mu_b, eps,
    sensor_a_letter, sensor_b_letter)."""
    data_root = os.path.join(dynamic_root, task)
    image_size = dd.get('image_size', 224)
    pair_mu = {}
    for pk in ('de', 'ef', 'fd'):
        ds, sa_name, sb_name = _build_pair_dataset(
            dd, data_root, pk, dd['pairs'][pk], image_size, [task], ef_stride)
        if len(ds) == 0:
            print(f"  [skip] {task}/{pk}: empty")
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        mu_a, mu_b, eps, act_a, act_b = [], [], [], [], []
        for (xa, xb), labels in loader:
            out = model.forward_pair(xa.to(device), xb.to(device), pk)
            mu_a.append(out['mu_a'].cpu().numpy())
            mu_b.append(out['mu_b'].cpu().numpy())
            eps.append(labels[1].cpu().numpy())            # episode_id
            # per-frame activity = std of the z-scored input tensor — a
            # contact proxy (high during contact, low at rest).
            act_a.append(xa.flatten(1).std(dim=1).numpy())
            act_b.append(xb.flatten(1).std(dim=1).numpy())
        pair_mu[pk] = (np.concatenate(mu_a).astype(np.float32),
                       np.concatenate(mu_b).astype(np.float32),
                       np.concatenate(eps),
                       _NAME2LETTER[sa_name], _NAME2LETTER[sb_name],
                       np.concatenate(act_a), np.concatenate(act_b))
        print(f"  {task}/{pk}: {pair_mu[pk][0].shape}")
    return pair_mu


def sensor_pools(pair_mu):
    """Pool latents per sensor from a task's pair_mu.
    Returns {sensor: (mu (N,d), ep_label (N,), ep_int (N,), act (N,))}."""
    pools = defaultdict(lambda: ([], [], [], []))
    for pk, (mu_a, mu_b, eps, sa, sb, act_a, act_b) in pair_mu.items():
        for mu, sensor, act in ((mu_a, sa, act_a), (mu_b, sb, act_b)):
            pools[sensor][0].append(mu)
            pools[sensor][1].append(np.array([f'{pk}e{e}' for e in eps]))
            pools[sensor][2].append(np.asarray(eps))
            pools[sensor][3].append(act)
    return {s: (np.concatenate(m), np.concatenate(l),
                np.concatenate(ei), np.concatenate(a))
            for s, (m, l, ei, a) in pools.items()}


def _tsne(x, perp, seed):
    p = min(perp, max(5, (len(x) - 1) // 3))
    return TSNE(n_components=2, perplexity=p, init='pca',
                random_state=seed).fit_transform(x)


def _max_contact_idx(eps, act, topk):
    """Per episode, indices of the top-`topk` highest-activity (max-contact)
    frames."""
    keep = []
    for e in np.unique(eps):
        idx = np.where(eps == e)[0]
        order = idx[np.argsort(act[idx])[::-1]]      # high activity first
        keep.extend(order[:topk].tolist())
    return keep


def _mean_unit(mu):
    """Mean of the L2-normalized rows of mu (centroid of unit vectors).
    dot(_mean_unit(A), _mean_unit(B)) == mean_{i,j} cos(A_i, B_j)."""
    nrm = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
    return nrm.mean(axis=0)


def _subsample(n, cap, seed):
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, cap, replace=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--weights', default='model.pt')
    parser.add_argument('--dynamic_root', default='latent_data_processed_dynamic')
    parser.add_argument('--datasets', nargs='+', default=['plug', 'pnp', 'wipe'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ef_stride', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_points', type=int, default=1200,
                        help='Cap per group fed to t-SNE.')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cross_task', action='store_true',
                        help='Also compute cross-task cosine similarity — '
                             'how similar one task\'s latents are to another '
                             'task\'s, per sensor (task x task matrix) -> '
                             '<out>_cross_task_cossim.png.')
    parser.add_argument('--max_contact_topk', type=int, default=10,
                        help='For --cross_task / --cross_sensor_task: per '
                             'episode, use just the top-K highest-activity '
                             '(max-contact) frames. Default 10. '
                             'A/B/C always use the full data.')
    parser.add_argument('--cross_sensor_task', action='store_true',
                        help='Also compute cross-SENSOR task x task cosine — '
                             'per sensor pair (de/ef/fd): cos(sensorA task-X '
                             'centroid, sensorB task-Y centroid) at max '
                             'contact. Diagonal = same-task cross-modal '
                             'alignment, off-diag = cross-task. -> '
                             '<out>_cross_sensor_task_cossim.png.')
    parser.add_argument('--out', default='dynamic3_latents')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dynamic_root = (args.dynamic_root if os.path.isabs(args.dynamic_root)
                    else os.path.join(repo_root, args.dynamic_root))

    # model + stats/pairs from the checkpoint's saved dataset_details
    with open(os.path.join(args.checkpoint_dir, 'dataset_details.yaml')) as f:
        dd = yaml.safe_load(f)
    model = _load_model(args.checkpoint_dir, args.weights, args.device)
    print(f"Loaded {args.checkpoint_dir}/{args.weights}\n")

    # ── encode all 3 datasets ──
    task_pair_mu = {}
    for task in args.datasets:
        print(f"=== {task} ===")
        task_pair_mu[task] = encode_task(task, dd, dynamic_root, model,
                                         args.device, args.ef_stride,
                                         args.batch_size)
    task_pools = {t: sensor_pools(pm) for t, pm in task_pair_mu.items()}

    out_base = (args.out if os.path.isabs(args.out)
                else os.path.join(repo_root, args.out))

    # ── A. combined t-SNE (colour = dataset, marker = sensor) ──
    groups = []                          # (dataset, sensor, mu)
    for task in args.datasets:
        for sensor, (mu, _lbl, _ei, _act) in task_pools.get(task, {}).items():
            idx = _subsample(len(mu), args.max_points, args.seed)
            groups.append((task, sensor, mu[idx]))
    if groups:
        sizes = [g[2].shape[0] for g in groups]
        emb = _tsne(np.concatenate([g[2] for g in groups]),
                    args.tsne_perplexity, args.seed)
        fig, ax = plt.subplots(figsize=(9, 8))
        off = 0
        for (task, sensor, mu), sz in zip(groups, sizes):
            pts = emb[off:off + sz]; off += sz
            ax.scatter(pts[:, 0], pts[:, 1], s=9, alpha=0.55,
                       c=_DATASET_COLOR.get(task, 'gray'),
                       marker=_SENSOR_MARKER[sensor], edgecolors='none')
        handles = [Line2D([], [], color=c, marker='o', ls='', label=t)
                   for t, c in _DATASET_COLOR.items() if t in args.datasets]
        handles += [Line2D([], [], color='gray', marker=m, ls='', label=s)
                    for s, m in _SENSOR_MARKER.items()]
        ax.legend(handles=handles, fontsize=8, ncol=2)
        ax.set_title('Dynamic latents — colour = dataset, marker = sensor')
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f'{out_base}_combined_tsne.png', dpi=150)
        plt.close(fig)
        print(f"saved {out_base}_combined_tsne.png")

    # ── B. per (dataset, sensor) individual t-SNE, coloured by episode ──
    fig, axes = plt.subplots(len(args.datasets), 3,
                             figsize=(14, 4.4 * len(args.datasets)),
                             squeeze=False)
    for r, task in enumerate(args.datasets):
        for c, sensor in enumerate(('D', 'E', 'F')):
            ax = axes[r][c]
            pool = task_pools.get(task, {}).get(sensor)
            if pool is None:
                ax.set_title(f'{task} / {sensor} (none)', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            mu, lbl = pool[0], pool[1]
            idx = _subsample(len(mu), args.max_points, args.seed)
            mu, lbl = mu[idx], lbl[idx]
            emb = _tsne(mu, args.tsne_perplexity, args.seed)
            uniq = sorted(set(lbl.tolist()))
            cmap = plt.cm.get_cmap('tab20', max(1, len(uniq)))
            for k, u in enumerate(uniq):
                m = lbl == u
                ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6,
                           color=cmap(k), edgecolors='none', label=u)
            ax.set_title(f'{task} / {sensor}  (n={len(mu)})', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=5, ncol=2, loc='best', framealpha=0.6)
    fig.suptitle('Per (dataset, sensor) t-SNE — coloured by <pair>e<episode>',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'{out_base}_per_obj_sensor_tsne.png', dpi=150)
    plt.close(fig)
    print(f"saved {out_base}_per_obj_sensor_tsne.png")

    # ── C. per-episode matched-pair cosine similarity (3x3) ──
    fig, axes = plt.subplots(len(args.datasets), 3,
                             figsize=(16, 4.0 * len(args.datasets)),
                             squeeze=False)
    print("\nPer-episode matched-pair cosine similarity:")
    for r, task in enumerate(args.datasets):
        pm = task_pair_mu.get(task, {})
        for c, pk in enumerate(('de', 'ef', 'fd')):
            ax = axes[r][c]
            if pk not in pm:
                ax.set_title(f'{task} / {pk} (none)', fontsize=8)
                continue
            mu_a, mu_b, eps, sa, sb, _act_a, _act_b = pm[pk]
            an = mu_a / (np.linalg.norm(mu_a, axis=1, keepdims=True) + 1e-8)
            bn = mu_b / (np.linalg.norm(mu_b, axis=1, keepdims=True) + 1e-8)
            cos = (an * bn).sum(axis=1)
            by_ep = defaultdict(list)
            for cv, e in zip(cos, eps):
                by_ep[int(e)].append(float(cv))
            keys = sorted(by_ep)
            means = [float(np.mean(by_ep[k])) for k in keys]
            stds = [float(np.std(by_ep[k])) for k in keys]
            x = np.arange(len(keys))
            ax.bar(x, means, yerr=stds, capsize=3,
                   color=_DATASET_COLOR.get(task, '#4C72B0'), alpha=0.85)
            ov = float(cos.mean())
            ax.axhline(ov, color='crimson', ls='--', lw=1.1,
                       label=f'mean {ov:+.3f}')
            ax.axhline(0, color='gray', lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels([f'ep{k}' for k in keys],
                                                 fontsize=7)
            ax.set_ylim(min(-0.1, min(means) - 0.15), 1.0)
            ax.set_title(f'{task} / {pk}  ({sa} vs {sb})', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            row = "  ".join(f'ep{k}={m:+.3f}' for k, m in zip(keys, means))
            print(f"  {task} {pk} ({sa}-{sb}): {row}  | mean {ov:+.3f}")
    fig.suptitle('Per-episode matched-pair cosine similarity', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'{out_base}_per_episode_cossim.png', dpi=150)
    plt.close(fig)
    print(f"\nsaved {out_base}_per_episode_cossim.png")

    # ── D. cross-task cosine similarity (flag) ──
    # Per sensor, mean pairwise cosine similarity between every pair of tasks'
    # latent sets. Equivalent to dot(mean of normalized A, mean of normalized
    # B) = mean_{i in A, j in B} cos(z_i, z_j). Diagonal = within-task
    # tightness; off-diagonal = how similar two tasks look in latent space.
    if args.cross_task:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
        axes = axes[0]
        print("\nCross-task mean cosine similarity (per sensor):")
        for ax, sensor in zip(axes, ('D', 'E', 'F')):
            tasks_present = [t for t in args.datasets
                             if sensor in task_pools.get(t, {})]
            cen = {}
            for t in tasks_present:
                mu, _lbl, ep_int, act = task_pools[t][sensor]
                # max-contact only: per episode keep the top-K frames by
                # activity (std of the input) — peak contact, not full traj.
                keep = _max_contact_idx(ep_int, act, args.max_contact_topk)
                cen[t] = _mean_unit(mu[keep])
            k = len(tasks_present)
            M = np.zeros((k, k), dtype=np.float32)
            for i, ta in enumerate(tasks_present):
                for j, tb in enumerate(tasks_present):
                    M[i, j] = float(cen[ta] @ cen[tb])
            im = ax.imshow(M, cmap='viridis', vmin=-1.0, vmax=1.0)
            ax.set_xticks(range(k)); ax.set_xticklabels(tasks_present, fontsize=8)
            ax.set_yticks(range(k)); ax.set_yticklabels(tasks_present, fontsize=8)
            for i in range(k):
                for j in range(k):
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                            color='white' if M[i, j] < 0.5 else 'black',
                            fontsize=9)
            ax.set_title(f'sensor {sensor}', fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
            print(f"  sensor {sensor}:")
            for i, ta in enumerate(tasks_present):
                row = "  ".join(f'{tb}={M[i, j]:+.3f}'
                                for j, tb in enumerate(tasks_present))
                print(f"    {ta:5s}: {row}")
        fig.suptitle('Cross-task mean pairwise cosine similarity '
                     f'(max-contact top-{args.max_contact_topk} frames/episode; '
                     'diag = within-task tightness)', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f'{out_base}_cross_task_cossim.png', dpi=150)
        plt.close(fig)
        print(f"\nsaved {out_base}_cross_task_cossim.png")

    # ── E. cross-SENSOR task x task cosine (flag) ──
    # Per sensor pair (de/ef/fd): M[X,Y] = cos(sensorA task-X centroid,
    # sensorB task-Y centroid), max-contact. Diagonal = same task, two sensors
    # = cross-modal alignment. Off-diagonal = different task, two sensors.
    # Matrix is asymmetric: rows = sensorA's task, cols = sensorB's task.
    if args.cross_sensor_task:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), squeeze=False)
        axes = axes[0]
        print("\nCross-sensor task x task cosine (per pair, max-contact):")
        for ax, pk in zip(axes, ('de', 'ef', 'fd')):
            cenA, cenB, sa, sb = {}, {}, None, None
            for task in args.datasets:
                pm = task_pair_mu.get(task, {}).get(pk)
                if pm is None:
                    continue
                mu_a, mu_b, eps, sa, sb, act_a, act_b = pm
                cenA[task] = _mean_unit(
                    mu_a[_max_contact_idx(eps, act_a, args.max_contact_topk)])
                cenB[task] = _mean_unit(
                    mu_b[_max_contact_idx(eps, act_b, args.max_contact_topk)])
            tasks = [t for t in args.datasets if t in cenA]
            if not tasks:
                ax.set_title(f'{pk} (none)', fontsize=9)
                continue
            k = len(tasks)
            M = np.zeros((k, k), dtype=np.float32)
            for i, X in enumerate(tasks):
                for j, Y in enumerate(tasks):
                    M[i, j] = float(cenA[X] @ cenB[Y])
            im = ax.imshow(M, cmap='viridis', vmin=-1.0, vmax=1.0)
            ax.set_yticks(range(k)); ax.set_yticklabels(tasks, fontsize=8)
            ax.set_xticks(range(k)); ax.set_xticklabels(tasks, fontsize=8)
            for i in range(k):
                for j in range(k):
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                            color='white' if M[i, j] < 0.5 else 'black',
                            fontsize=9)
            ax.set_ylabel(f'{sa} task'); ax.set_xlabel(f'{sb} task')
            ax.set_title(f'{pk}:  {sa} vs {sb}', fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
            print(f"  {pk} ({sa} vs {sb}):  rows={sa}-task, cols={sb}-task")
            for i, X in enumerate(tasks):
                row = "  ".join(f'{Y}={M[i, j]:+.3f}'
                                for j, Y in enumerate(tasks))
                print(f"    {sa}:{X:5s} -> {row}")
        fig.suptitle('Cross-sensor task x task cosine (max-contact) — '
                     'diagonal = same-task cross-modal alignment', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f'{out_base}_cross_sensor_task_cossim.png', dpi=150)
        plt.close(fig)
        print(f"\nsaved {out_base}_cross_sensor_task_cossim.png")


if __name__ == '__main__':
    main()
