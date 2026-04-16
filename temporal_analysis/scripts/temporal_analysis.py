"""
Temporal analysis of cross-modal embeddings within episodes.

Analyzes:
  1. Embedding trajectory smoothness — cosine sim between consecutive frames (t, t+1)
  2. Cross-modal alignment over time — cosine sim between Daimon[t] and eFlesh[t] (same timestamp)
     vs Daimon[t] and eFlesh[t'] (different timestamp within same episode)
  3. Embedding drift — how far each frame is from the episode mean
  4. Per-episode trajectory plots in 2D (PCA) showing temporal flow

Produces plots for both train and val splits.

Usage:
  PYTHONPATH=. python scripts/temporal_analysis.py --run_name dataset_cross_modal_run_cross_modal_z64_20260415_...
  PYTHONPATH=. python scripts/temporal_analysis.py --run_name dataset_cross_modal_run_cross_modal_z64_20260415_... --n_episodes 5
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.utils.utils import get_config_details
from joint_embedding_learning.utils.models_setup import get_trained_model
from joint_embedding_learning.data.datasets import get_all_contrastive_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal analysis of cross-modal embeddings")
    parser.add_argument("--dataset_name", type=str, default="dataset_cross_modal")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoint_root", type=str, default=None,
                        help="Root checkpoints dir. Defaults to <project>/checkpoints/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=4,
                        help="Number of episodes to plot individual trajectories for")
    parser.add_argument("--frame_gaps", type=int, nargs="+", default=[5, 15, 30],
                        help="Frame gaps k to compare cos(z[t], z[t+k]). Default: [5, 15, 30] at 25Hz = 0.2s, 0.6s, 1.2s")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ===== Extract per-episode embeddings =====

def get_base_dataset(ds):
    """Unwrap ConcatDataset / Subset to get underlying CrossModalDataset."""
    if hasattr(ds, 'dataset'):
        return ds.dataset
    return ds


def extract_episode_info(datasets):
    """Extract (episode_id, frame_idx) for each sample from filenames."""
    episodes = []
    frame_indices = []
    for ds in datasets:
        base_ds = get_base_dataset(ds)
        for f in base_ds.daimon_files:
            fname = os.path.basename(f)  # ep0003_fr0042.pt
            parts = fname.replace(".pt", "").split("_")
            ep = int(parts[0].replace("ep", ""))
            fr = int(parts[1].replace("fr", ""))
            episodes.append(ep)
            frame_indices.append(fr)
    return np.array(episodes), np.array(frame_indices)


@torch.no_grad()
def extract_embeddings(loader, model, device):
    """Extract h_i, h_j, z_i, z_j for all data."""
    all_h_i, all_h_j, all_z_i, all_z_j = [], [], [], []
    for (x_i, x_j), _ in tqdm(loader, desc="Extracting"):
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        all_h_i.append(h_i.cpu())
        all_h_j.append(h_j.cpu())
        all_z_i.append(z_i.cpu())
        all_z_j.append(z_j.cpu())
    return (torch.cat(all_h_i), torch.cat(all_h_j),
            torch.cat(all_z_i), torch.cat(all_z_j))


def group_by_episode(episodes, frame_indices, *tensors):
    """Group tensors by episode, sorted by frame index within each episode."""
    unique_eps = np.unique(episodes)
    grouped = {}
    for ep in unique_eps:
        mask = episodes == ep
        order = np.argsort(frame_indices[mask])
        grouped[ep] = {
            'frame_indices': frame_indices[mask][order],
            'tensors': [t[mask][order] for t in tensors],
        }
    return grouped


# ===== Analysis functions =====

def compute_gap_similarity(z_seq, k):
    """Cosine similarity between frames k apart: sim(z[t], z[t+k])."""
    if len(z_seq) <= k:
        return np.array([])
    z_norm = F.normalize(z_seq, dim=1)
    sims = (z_norm[:-k] * z_norm[k:]).sum(dim=1).numpy()
    return sims


def compute_cross_modal_temporal_similarity(z_i_seq, z_j_seq):
    """
    For each frame t:
      - same_t: cos(z_i[t], z_j[t])  — same timestamp, cross-modal
      - diff_t: mean cos(z_i[t], z_j[t']) for t' != t — different timestamp, cross-modal
    """
    z_i_norm = F.normalize(z_i_seq, dim=1)
    z_j_norm = F.normalize(z_j_seq, dim=1)

    # Full similarity matrix: (T, T)
    sim_matrix = (z_i_norm @ z_j_norm.T).numpy()

    same_t = np.diag(sim_matrix)

    # Mean off-diagonal per row
    T = len(sim_matrix)
    if T > 1:
        mask = ~np.eye(T, dtype=bool)
        diff_t = (sim_matrix * mask).sum(axis=1) / (T - 1)
    else:
        diff_t = np.zeros(T)

    return same_t, diff_t, sim_matrix


# ===== Plotting =====

def plot_gap_smoothness(sims_by_gap, split_name, save_path, fps=25):
    """Histograms of cos(z[t], z[t+k]) for multiple gaps k, per modality.

    sims_by_gap: dict {k: {'daimon': [arr_per_ep,...], 'eflesh': [arr_per_ep,...]}}
    """
    gaps = sorted(sims_by_gap.keys())
    n_gaps = len(gaps)
    fig, axes = plt.subplots(2, n_gaps, figsize=(5 * n_gaps, 9), squeeze=False)
    fig.suptitle(f"Frame Gap Similarity — {split_name}", fontsize=15, fontweight="bold")

    for col, k in enumerate(gaps):
        for row, (modality, color) in enumerate([("daimon", "tab:blue"), ("eflesh", "tab:orange")]):
            per_ep = sims_by_gap[k][modality]
            per_ep = [a for a in per_ep if len(a) > 0]
            if not per_ep:
                continue
            all_sims = np.concatenate(per_ep)
            ax = axes[row, col]
            ax.hist(all_sims, bins=60, alpha=0.7, color=color, density=True)
            ax.axvline(x=all_sims.mean(), color="red", linestyle="--", linewidth=1.5)
            ax.set_xlabel(f"cos(z[t], z[t+{k}])", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            sec = k / fps
            ax.set_title(f"{modality.capitalize()} — gap {k} frames ({sec:.2f}s)\nmean={all_sims.mean():.3f}, std={all_sims.std():.3f}", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cross_modal_temporal(all_same_t, all_diff_t, split_name, save_path):
    """Same-timestamp vs different-timestamp cross-modal similarity."""
    same = np.concatenate(all_same_t)
    diff = np.concatenate(all_diff_t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cross-Modal Temporal Alignment — {split_name}", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.hist(same, bins=80, alpha=0.7, color="tab:green", density=True, label=f"Same t (mean={same.mean():.3f})")
    ax.hist(diff, bins=80, alpha=0.7, color="tab:red", density=True, label=f"Diff t (mean={diff.mean():.3f})")
    ax.axvline(x=same.mean(), color="green", linestyle="--", alpha=0.7)
    ax.axvline(x=diff.mean(), color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("cos(Daimon[t], eFlesh[t]) vs cos(Daimon[t], eFlesh[t'])", fontsize=12)
    ax.legend(fontsize=10)

    # Gap over time within episodes (pick longest episode)
    ax = axes[1]
    gap_per_frame = [s - d for s, d in zip(all_same_t, all_diff_t)]
    # Show a few episodes
    for i, gap in enumerate(gap_per_frame[:5]):
        ax.plot(gap, alpha=0.6, linewidth=1, label=f"ep {i}")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Frame within episode", fontsize=11)
    ax.set_ylabel("Same_t - Diff_t gap", fontsize=11)
    ax.set_title("Cross-modal alignment gap over time", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_similarity_matrices(grouped, ep_ids, space_name, save_path):
    """Plot cross-modal similarity matrices for selected episodes."""
    n = len(ep_ids)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Cross-Modal Sim Matrix ({space_name}) — Daimon[row] vs eFlesh[col]", fontsize=14, fontweight="bold")

    for ax, ep_id in zip(axes, ep_ids):
        data = grouped[ep_id]
        z_i, z_j = data['tensors'][0], data['tensors'][1]
        _, _, sim_matrix = compute_cross_modal_temporal_similarity(z_i, z_j)

        im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xlabel("eFlesh frame", fontsize=10)
        ax.set_ylabel("Daimon frame", fontsize=10)
        ax.set_title(f"Episode {ep_id} ({len(z_i)} frames)", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_trajectory_pca(grouped, ep_ids, space_name, save_path):
    """PCA trajectory plots for selected episodes — shows temporal flow."""
    # Fit PCA on all data from selected episodes
    all_z = []
    for ep_id in ep_ids:
        z_i, z_j = grouped[ep_id]['tensors'][0], grouped[ep_id]['tensors'][1]
        all_z.append(z_i)
        all_z.append(z_j)
    all_z = torch.cat(all_z).numpy()
    pca = PCA(n_components=2)
    pca.fit(all_z)

    n = len(ep_ids)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Embedding Trajectory (PCA, {space_name})", fontsize=14, fontweight="bold")

    for ax, ep_id in zip(axes, ep_ids):
        z_i, z_j = grouped[ep_id]['tensors'][0], grouped[ep_id]['tensors'][1]
        T = len(z_i)

        pca_i = pca.transform(z_i.numpy())
        pca_j = pca.transform(z_j.numpy())

        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, T))

        # Daimon trajectory
        ax.scatter(pca_i[:, 0], pca_i[:, 1], c=colors, s=20, marker="o", alpha=0.7, zorder=2)
        ax.plot(pca_i[:, 0], pca_i[:, 1], c="tab:blue", alpha=0.3, linewidth=1, zorder=1)

        # eFlesh trajectory
        ax.scatter(pca_j[:, 0], pca_j[:, 1], c=colors, s=20, marker="x", alpha=0.7, zorder=2)
        ax.plot(pca_j[:, 0], pca_j[:, 1], c="tab:orange", alpha=0.3, linewidth=1, zorder=1)

        # Connect same-timestamp pairs (subsample for clarity)
        step = max(1, T // 20)
        for t in range(0, T, step):
            ax.plot([pca_i[t, 0], pca_j[t, 0]], [pca_i[t, 1], pca_j[t, 1]],
                    c=colors[t], alpha=0.4, linewidth=0.8, zorder=0)

        # Mark start and end for both modalities
        ax.scatter(pca_i[0, 0], pca_i[0, 1], facecolors="none", edgecolors="green", s=180,
                   marker="o", linewidths=2.5, zorder=3, label="Daimon Start")
        ax.scatter(pca_i[-1, 0], pca_i[-1, 1], facecolors="none", edgecolors="red", s=180,
                   marker="o", linewidths=2.5, zorder=3, label="Daimon End")
        ax.scatter(pca_j[0, 0], pca_j[0, 1], c="green", s=120,
                   marker="x", linewidths=3, zorder=3, label="eFlesh Start")
        ax.scatter(pca_j[-1, 0], pca_j[-1, 1], c="red", s=120,
                   marker="x", linewidths=3, zorder=3, label="eFlesh End")

        ax.set_title(f"Ep {ep_id} ({T} frames) — o=Daimon, x=eFlesh", fontsize=11)
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def analyze_split(datasets, loader, model, device, split_name, output_dir, n_episodes, rng, frame_gaps):
    """Run full temporal analysis on one split."""
    print(f"\n{'='*60}")
    print(f"  {split_name}")
    print(f"{'='*60}")

    episodes, frame_indices = extract_episode_info(datasets)
    h_i, h_j, z_i, z_j = extract_embeddings(loader, model, device)

    print(f"  Samples: {len(z_i)}, Episodes: {len(np.unique(episodes))}")
    print(f"  z dim: {z_i.shape[1]}, h dim: {h_i.shape[1]}")

    # Group by episode (z space)
    grouped_z = group_by_episode(episodes, frame_indices, z_i, z_j)

    # --- 1. Frame-gap smoothness (z space) ---
    sims_by_gap = {k: {'daimon': [], 'eflesh': []} for k in frame_gaps}
    for ep_id in sorted(grouped_z.keys()):
        zi, zj = grouped_z[ep_id]['tensors']
        for k in frame_gaps:
            sims_by_gap[k]['daimon'].append(compute_gap_similarity(zi, k))
            sims_by_gap[k]['eflesh'].append(compute_gap_similarity(zj, k))

    plot_gap_smoothness(sims_by_gap, split_name,
                        os.path.join(output_dir, f"frame_gap_smoothness_{split_name.lower()}.png"))

    gap_stats = {}
    for k in frame_gaps:
        d = np.concatenate([a for a in sims_by_gap[k]['daimon'] if len(a) > 0])
        e = np.concatenate([a for a in sims_by_gap[k]['eflesh'] if len(a) > 0])
        gap_stats[k] = (d.mean(), e.mean())
        print(f"  Gap {k} ({k/25:.2f}s): Daimon={d.mean():.4f}, eFlesh={e.mean():.4f}")

    # --- 2. Cross-modal: same-t vs diff-t ---
    all_same_t, all_diff_t = [], []
    for ep_id in sorted(grouped_z.keys()):
        zi, zj = grouped_z[ep_id]['tensors']
        if len(zi) < 2:
            continue
        same_t, diff_t, _ = compute_cross_modal_temporal_similarity(zi, zj)
        all_same_t.append(same_t)
        all_diff_t.append(diff_t)

    plot_cross_modal_temporal(
        all_same_t, all_diff_t, split_name,
        os.path.join(output_dir, f"cross_modal_temporal_{split_name.lower()}.png"))

    same_all = np.concatenate(all_same_t)
    diff_all = np.concatenate(all_diff_t)
    print(f"  Cross-modal sim (z): same_t={same_all.mean():.4f}, diff_t={diff_all.mean():.4f}, gap={same_all.mean() - diff_all.mean():.4f}")

    # --- 3. Pick episodes for detailed plots ---
    ep_ids_sorted = sorted(grouped_z.keys(), key=lambda e: len(grouped_z[e]['tensors'][0]), reverse=True)
    # Pick from among the longer episodes
    candidates = ep_ids_sorted[:max(10, len(ep_ids_sorted))]
    selected = sorted(rng.choice(candidates, size=min(n_episodes, len(candidates)), replace=False))
    print(f"  Selected episodes for detail plots: {selected}")

    # Similarity matrices
    plot_similarity_matrices(
        grouped_z, selected, "z space",
        os.path.join(output_dir, f"sim_matrix_{split_name.lower()}.png"))

    # PCA trajectories
    plot_trajectory_pca(
        grouped_z, selected, "z space",
        os.path.join(output_dir, f"trajectory_pca_{split_name.lower()}.png"))

    out = {
        'cross_same_t': same_all.mean(),
        'cross_diff_t': diff_all.mean(),
        'cross_gap': same_all.mean() - diff_all.mean(),
    }
    for k, (d, e) in gap_stats.items():
        out[f'gap{k}_daimon'] = d
        out[f'gap{k}_eflesh'] = e
    return out


# ===== Main =====

def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_config = os.path.join(root, "config", "datasets_config.yaml")
    model_config = os.path.join(root, "config", "models_config.yaml")

    dataset_details = get_config_details(dataset_config, args.dataset_name)
    dataset_type = dataset_details["dataset_type"]

    # Resolve checkpoint root
    ckpt_root = args.checkpoint_root if args.checkpoint_root else os.path.join(root, "checkpoints")
    checkpoint_dir = os.path.join(ckpt_root, dataset_type, "cross_modal", args.run_name)

    # Load model details from checkpoint yaml (matches what was actually trained)
    import yaml
    ckpt_model_yaml = os.path.join(checkpoint_dir, "model_details.yaml")
    if os.path.exists(ckpt_model_yaml):
        with open(ckpt_model_yaml, 'r') as f:
            model_details = yaml.safe_load(f)
    else:
        model_details = get_config_details(model_config, "cross_modal")

    if args.output_dir is None:
        args.output_dir = os.path.join(checkpoint_dir, "temporal")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Temporal Analysis — Cross-Modal Embeddings")
    print("=" * 60)
    print(f"  Run name       : {args.run_name}")
    print(f"  Checkpoint     : {checkpoint_dir}")
    print(f"  Projection dim : {model_details['projection_dim']}")
    print(f"  Device         : {args.device}")
    print(f"  Output         : {args.output_dir}")

    # Load model from checkpoint
    model_fp = os.path.join(checkpoint_dir, "model.pt")
    from joint_embedding_learning.utils.models_setup import get_model
    model = get_model(model_details)
    model.load_state_dict(torch.load(model_fp, map_location=args.device))
    model = model.to(args.device)
    model.eval()

    # Load data
    from torch.utils.data import ConcatDataset, DataLoader
    train_datasets, val_datasets, _ = get_all_contrastive_datasets(args.dataset_name, data_split="test", device="cpu")

    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")

    # Analyze both splits
    train_stats = analyze_split(train_datasets, train_loader, model, args.device,
                                "Train", args.output_dir, args.n_episodes, rng, args.frame_gaps)
    val_stats = analyze_split(val_datasets, val_loader, model, args.device,
                              "Val", args.output_dir, args.n_episodes, rng, args.frame_gaps)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  {'Metric':<30s} {'Train':>10s} {'Val':>10s}")
    print(f"  {'-'*50}")
    for key in train_stats:
        print(f"  {key:<30s} {train_stats[key]:>10.4f} {val_stats[key]:>10.4f}")
    print(f"  Plots saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
