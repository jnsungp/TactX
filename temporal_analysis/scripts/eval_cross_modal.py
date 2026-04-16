"""
Evaluate a trained CrossModalCLR checkpoint (Daimon + eFlesh).

Produces:
  1. t-SNE plots (projection space z) for train and val — modality, episode, pair links
  2. Object-level t-SNE (2x3 grid) — per-object, per-modality, cosine sim matrix
  3. Cosine similarity histograms (positive vs negative pairs) in z and h space
  4. Cross-modal retrieval accuracy (Top-k)

Usage:
  python scripts/eval_cross_modal.py --run_name dataset_cross_modal_run_20260415_123456
  python scripts/eval_cross_modal.py --run_name dataset_cross_modal_run_20260415_123456 --device cuda:1
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
from sklearn.manifold import TSNE
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.utils.utils import get_config_details
from joint_embedding_learning.utils.models_setup import get_trained_model
from joint_embedding_learning.data.datasets import get_all_contrastive_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CrossModalCLR checkpoint")
    parser.add_argument("--dataset_name", type=str, default="dataset_cross_modal")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Checkpoint run name (e.g. dataset_cross_modal_run_20260415_123456)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots. Defaults to checkpoints/<type>/<model>/<run_name>/eval/")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--tsne_max_iter", type=int, default=1000)
    return parser.parse_args()


# ===== Helpers =====

def get_base_dataset(ds):
    """Unwrap ConcatDataset / Subset to get the underlying CrossModalDataset."""
    if hasattr(ds, 'dataset'):
        return ds.dataset
    return ds


def extract_episode_ids(datasets):
    """Extract episode IDs from daimon filenames (e.g. ep0003_fr0042.pt -> 3)."""
    episode_ids = []
    for ds in datasets:
        base_ds = get_base_dataset(ds)
        for f in base_ds.daimon_files:
            ep = int(os.path.basename(f).split("_")[0].replace("ep", ""))
            episode_ids.append(ep)
    return np.array(episode_ids)


# ===== Embedding extraction =====

@torch.no_grad()
def extract_cross_modal_embeddings(loader, model, device):
    """Extract h_i, h_j, z_i, z_j, labels from CrossModalCLR."""
    all_h_i, all_h_j, all_z_i, all_z_j, all_labels = [], [], [], [], []
    for (x_i, x_j), labels in tqdm(loader, desc="Extracting"):
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        all_h_i.append(h_i.cpu())
        all_h_j.append(h_j.cpu())
        all_z_i.append(z_i.cpu())
        all_z_j.append(z_j.cpu())
        all_labels.append(labels[0].cpu())
    return (torch.cat(all_h_i), torch.cat(all_h_j),
            torch.cat(all_z_i), torch.cat(all_z_j),
            torch.cat(all_labels))


# ===== t-SNE with episode coloring (notebook cells 3 & 4) =====

def plot_tsne_episode(z_i, z_j, episode_ids, split_name, save_path,
                      perplexity=30, max_iter=1000):
    """t-SNE: modality coloring, episode coloring, positive pair links."""
    N = len(z_i)
    idx = np.arange(N)

    z_i_np = F.normalize(z_i[idx], dim=1).numpy()
    z_j_np = F.normalize(z_j[idx], dim=1).numpy()
    combined = np.concatenate([z_i_np, z_j_np], axis=0)

    print(f"[{split_name}] Running t-SNE on {combined.shape[0]} points ({N} per modality)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=max_iter)
    tsne_2d = tsne.fit_transform(combined)

    tsne_daimon = tsne_2d[:N]
    tsne_eflesh = tsne_2d[N:]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # --- Plot 1: modality ---
    ax = axes[0]
    ax.scatter(tsne_daimon[:, 0], tsne_daimon[:, 1], c="tab:blue", alpha=0.4, s=15, label="Daimon (image)")
    ax.scatter(tsne_eflesh[:, 0], tsne_eflesh[:, 1], c="tab:orange", alpha=0.4, s=15, label="eFlesh (magnetic)")
    ax.set_title("t-SNE: Modality", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # --- Plot 2: episode ---
    ax = axes[1]
    unique_eps = np.unique(episode_ids[idx])
    cmap = plt.cm.get_cmap("tab20", len(unique_eps))
    for i, ep in enumerate(unique_eps):
        mask = episode_ids[idx] == ep
        ax.scatter(tsne_daimon[mask, 0], tsne_daimon[mask, 1], c=[cmap(i)], alpha=0.5, s=15, marker="o")
        ax.scatter(tsne_eflesh[mask, 0], tsne_eflesh[mask, 1], c=[cmap(i)], alpha=0.5, s=15, marker="x")
    ax.set_title("t-SNE: Episode (o=Daimon, x=eFlesh)", fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])

    # --- Plot 3: positive pair links ---
    ax = axes[2]
    ax.scatter(tsne_daimon[:, 0], tsne_daimon[:, 1], c="tab:blue", alpha=0.3, s=10, label="Daimon")
    ax.scatter(tsne_eflesh[:, 0], tsne_eflesh[:, 1], c="tab:orange", alpha=0.3, s=10, label="eFlesh")
    sample_pairs = np.random.RandomState(0).choice(N, size=min(200, N), replace=False)
    for k in sample_pairs:
        ax.plot([tsne_daimon[k, 0], tsne_eflesh[k, 0]],
                [tsne_daimon[k, 1], tsne_eflesh[k, 1]], c="gray", alpha=0.15, linewidth=0.5)
    ax.set_title("t-SNE: Positive pair links", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ===== Object-level t-SNE (notebook cell 5) =====

def plot_object_tsne(z_i, z_j, labels, tool_names, split_name, save_path,
                     perplexity=30, max_iter=1000):
    """2x3 t-SNE grid: per-object, per-modality, cosine sim matrix."""
    labels_np = labels.numpy()
    N = len(z_i)
    n_obj = len(tool_names)

    # Generate colors for arbitrary number of objects
    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                   "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                   "tab:olive", "tab:cyan"]
    object_colors = [base_colors[i % len(base_colors)] for i in range(n_obj)]

    z_i_np = F.normalize(z_i, dim=1).numpy()
    z_j_np = F.normalize(z_j, dim=1).numpy()
    combined = np.concatenate([z_i_np, z_j_np], axis=0)

    print(f"[{split_name}] Running object t-SNE on {combined.shape[0]} points...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=max_iter)
    tsne_2d = tsne.fit_transform(combined)

    tsne_d = tsne_2d[:N]
    tsne_e = tsne_2d[N:]

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f"t-SNE by Object — {split_name}", fontsize=16, fontweight="bold")

    # ---- Row 0, Col 0: both modalities colored by object ----
    for oi, (tname, color) in enumerate(zip(tool_names, object_colors)):
        mask = labels_np == oi
        axes[0, 0].scatter(tsne_d[mask, 0], tsne_d[mask, 1],
                           c=color, alpha=0.4, s=15, marker="o", label=f"{tname}")
        axes[0, 0].scatter(tsne_e[mask, 0], tsne_e[mask, 1],
                           c=color, alpha=0.4, s=20, marker="x")
    axes[0, 0].set_title("Object (o=Daimon, x=eFlesh)", fontsize=13)
    axes[0, 0].legend(fontsize=10, markerscale=1.5)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    # ---- Row 0, Col 1: modality coloring ----
    axes[0, 1].scatter(tsne_d[:, 0], tsne_d[:, 1], c="tab:blue", alpha=0.35, s=12, label="Daimon")
    axes[0, 1].scatter(tsne_e[:, 0], tsne_e[:, 1], c="tab:red", alpha=0.35, s=12, label="eFlesh")
    axes[0, 1].set_title("Modality", fontsize=13)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

    # ---- Row 0, Col 2: positive pair links colored by object ----
    axes[0, 2].scatter(tsne_d[:, 0], tsne_d[:, 1], c="tab:blue", alpha=0.25, s=10, label="Daimon")
    axes[0, 2].scatter(tsne_e[:, 0], tsne_e[:, 1], c="tab:red", alpha=0.25, s=10, label="eFlesh")
    sample_k = np.random.RandomState(0).choice(N, size=min(300, N), replace=False)
    for k in sample_k:
        color = object_colors[labels_np[k]]
        axes[0, 2].plot([tsne_d[k, 0], tsne_e[k, 0]],
                        [tsne_d[k, 1], tsne_e[k, 1]], c=color, alpha=0.2, linewidth=0.6)
    axes[0, 2].set_title("Positive pair links (colored by object)", fontsize=13)
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])

    # ---- Row 1, Col 0 & 1: per-modality by object ----
    for col, (tsne_pts, modal_name) in enumerate([(tsne_d, "Daimon"), (tsne_e, "eFlesh")]):
        ax = axes[1, col]
        for oi, (tname, color) in enumerate(zip(tool_names, object_colors)):
            mask = labels_np == oi
            ax.scatter(tsne_pts[mask, 0], tsne_pts[mask, 1],
                       c=color, alpha=0.5, s=18, label=tname)
        ax.set_title(f"{modal_name} — by Object", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # ---- Row 1, Col 2: inter-object cosine similarity matrix ----
    ax = axes[1, 2]
    sim_matrix = np.zeros((n_obj, n_obj))
    z_i_n = F.normalize(z_i, dim=1)
    z_j_n = F.normalize(z_j, dim=1)
    for oi in range(n_obj):
        for oj in range(n_obj):
            mask_i = labels_np == oi
            mask_j = labels_np == oj
            if mask_i.sum() == 0 or mask_j.sum() == 0:
                continue
            sim = (z_i_n[mask_i].mean(0) * z_j_n[mask_j].mean(0)).sum().item()
            sim_matrix[oi, oj] = sim
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n_obj)); ax.set_xticklabels(tool_names, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(n_obj)); ax.set_yticklabels(tool_names, fontsize=10)
    for oi in range(n_obj):
        for oj in range(n_obj):
            ax.text(oj, oi, f"{sim_matrix[oi, oj]:.3f}", ha="center", va="center", fontsize=11,
                    color="black" if abs(sim_matrix[oi, oj]) < 0.6 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Mean Cosine Sim: Daimon(row) vs eFlesh(col)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ===== Cosine similarity (notebook cell 6) =====

def plot_cosine_similarity(h_i, h_j, z_i, z_j, save_path):
    """Cosine similarity histograms: positive vs negative pairs in z and h space."""
    perm = np.random.RandomState(0).permutation(len(z_j))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # z space
    z_i_norm = F.normalize(z_i, dim=1)
    z_j_norm = F.normalize(z_j, dim=1)
    pos_sim_z = (z_i_norm * z_j_norm).sum(dim=1).numpy()
    neg_sim_z = (z_i_norm * z_j_norm[perm]).sum(dim=1).numpy()

    ax = axes[0]
    ax.hist(pos_sim_z, bins=80, alpha=0.7, label=f"Positive (mean={pos_sim_z.mean():.3f})", color="tab:green", density=True)
    ax.hist(neg_sim_z, bins=80, alpha=0.7, label=f"Negative (mean={neg_sim_z.mean():.3f})", color="tab:red", density=True)
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Val: Positive vs Negative Pair Similarity (z space)", fontsize=13)
    ax.legend(fontsize=11)
    ax.axvline(x=pos_sim_z.mean(), color="green", linestyle="--", alpha=0.7)
    ax.axvline(x=neg_sim_z.mean(), color="red", linestyle="--", alpha=0.7)

    # h space
    h_i_norm = F.normalize(h_i, dim=1)
    h_j_norm = F.normalize(h_j, dim=1)
    pos_sim_h = (h_i_norm * h_j_norm).sum(dim=1).numpy()
    neg_sim_h = (h_i_norm * h_j_norm[perm]).sum(dim=1).numpy()

    ax = axes[1]
    ax.hist(pos_sim_h, bins=80, alpha=0.7, label=f"Positive (mean={pos_sim_h.mean():.3f})", color="tab:green", density=True)
    ax.hist(neg_sim_h, bins=80, alpha=0.7, label=f"Negative (mean={neg_sim_h.mean():.3f})", color="tab:red", density=True)
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Val: Positive vs Negative Pair Similarity (h space)", fontsize=13)
    ax.legend(fontsize=11)
    ax.axvline(x=pos_sim_h.mean(), color="green", linestyle="--", alpha=0.7)
    ax.axvline(x=neg_sim_h.mean(), color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    return pos_sim_z.mean(), neg_sim_z.mean(), pos_sim_h.mean(), neg_sim_h.mean()


# ===== Retrieval accuracy (notebook cell 7) =====

def retrieval_accuracy(z_query, z_gallery, ks=(1, 5, 10)):
    """Top-k retrieval: query modality -> gallery modality."""
    z_q = F.normalize(z_query, dim=1)
    z_g = F.normalize(z_gallery, dim=1)
    sim = z_q @ z_g.T
    results = {}
    for k in ks:
        topk_idx = sim.topk(k, dim=1).indices
        correct = (topk_idx == torch.arange(len(z_q)).unsqueeze(1)).any(dim=1)
        results[k] = correct.float().mean().item()
    return results


# ===== Main =====

def main():
    args = parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_config = os.path.join(root, "config", "datasets_config.yaml")
    model_config = os.path.join(root, "config", "models_config.yaml")

    dataset_details = get_config_details(dataset_config, args.dataset_name)
    dataset_type = dataset_details["dataset_type"]

    # Load model details from checkpoint yaml (matches what was actually trained)
    import yaml
    checkpoint_dir = os.path.join(root, "checkpoints", dataset_type, "cross_modal", args.run_name)
    ckpt_model_yaml = os.path.join(checkpoint_dir, "model_details.yaml")
    if os.path.exists(ckpt_model_yaml):
        with open(ckpt_model_yaml, 'r') as f:
            model_details = yaml.safe_load(f)
    else:
        model_details = get_config_details(model_config, "cross_modal")

    train_tool_names = dataset_details["tools"]["training_tools"]
    val_tool_names = dataset_details["tools"].get("val_tools", [])

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(root, "checkpoints", dataset_type, "cross_modal", args.run_name, "eval")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Cross-Modal Evaluation")
    print("=" * 60)
    print(f"  Run name       : {args.run_name}")
    print(f"  Dataset        : {args.dataset_name}")
    print(f"  Projection dim : {model_details['projection_dim']}")
    print(f"  Train tools    : {train_tool_names}")
    print(f"  Val tools      : {val_tool_names}")
    print(f"  Device         : {args.device}")
    print(f"  Output         : {args.output_dir}")
    print()

    # --- Load model ---
    model = get_trained_model(model_details, dataset_type, args.run_name, args.device)
    model.eval()

    # --- Load data ---
    train_datasets, val_datasets, _ = get_all_contrastive_datasets(args.dataset_name, data_split="test", device="cpu")
    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"  Train samples  : {len(train_set)}")
    print(f"  Val samples    : {len(val_set)}")
    print()

    # --- Extract episode IDs from filenames ---
    train_episode_ids = extract_episode_ids(train_datasets)
    val_episode_ids = extract_episode_ids(val_datasets)

    # --- Extract embeddings ---
    print("Extracting embeddings...")
    train_h_i, train_h_j, train_z_i, train_z_j, train_labels = extract_cross_modal_embeddings(train_loader, model, args.device)
    val_h_i, val_h_j, val_z_i, val_z_j, val_labels = extract_cross_modal_embeddings(val_loader, model, args.device)
    print(f"  Train — h: {train_h_i.shape}, z: {train_z_i.shape}, labels: {train_labels.shape}")
    print(f"  Val   — h: {val_h_i.shape}, z: {val_z_i.shape}, labels: {val_labels.shape}")
    print()

    # --- t-SNE with episode coloring (notebook cells 3 & 4) ---
    print("Generating t-SNE plots (episode coloring)...")
    plot_tsne_episode(train_z_i, train_z_j, train_episode_ids, "Train",
                      os.path.join(args.output_dir, "tsne_cross_modal_train.png"),
                      perplexity=args.perplexity, max_iter=args.tsne_max_iter)
    plot_tsne_episode(val_z_i, val_z_j, val_episode_ids, "Val",
                      os.path.join(args.output_dir, "tsne_cross_modal_val.png"),
                      perplexity=args.perplexity, max_iter=args.tsne_max_iter)
    print()

    # --- Object-level t-SNE (notebook cell 5) ---
    if len(train_tool_names) > 1:
        print("Generating object-level t-SNE plots...")
        plot_object_tsne(train_z_i, train_z_j, train_labels, train_tool_names, "Train",
                         os.path.join(args.output_dir, "tsne_object_train.png"),
                         perplexity=args.perplexity, max_iter=args.tsne_max_iter)
        if val_tool_names:
            plot_object_tsne(val_z_i, val_z_j, val_labels, val_tool_names, "Val",
                             os.path.join(args.output_dir, "tsne_object_val.png"),
                             perplexity=args.perplexity, max_iter=args.tsne_max_iter)
        print()
    else:
        print("Skipping object-level t-SNE (only 1 training tool).")
        print()

    # --- Cosine similarity (notebook cell 6) ---
    print("Computing cosine similarity...")
    pos_z, neg_z, pos_h, neg_h = plot_cosine_similarity(
        val_h_i, val_h_j, val_z_i, val_z_j,
        os.path.join(args.output_dir, "cosine_similarity_cross_modal.png"))
    print(f"  [z space] Pos mean: {pos_z:.4f} | Neg mean: {neg_z:.4f} | Gap: {pos_z - neg_z:.4f}")
    print(f"  [h space] Pos mean: {pos_h:.4f} | Neg mean: {neg_h:.4f} | Gap: {pos_h - neg_h:.4f}")
    print()

    # --- Retrieval accuracy (notebook cell 7) ---
    print("===== Retrieval Accuracy (z space) =====")
    ks = (1, 5, 10)
    d2e_z = retrieval_accuracy(val_z_i, val_z_j, ks)
    e2d_z = retrieval_accuracy(val_z_j, val_z_i, ks)
    for k in ks:
        print(f"  Top-{k}: Daimon->eFlesh {d2e_z[k]*100:.1f}% | eFlesh->Daimon {e2d_z[k]*100:.1f}%")

    print("\n===== Retrieval Accuracy (h space) =====")
    d2e_h = retrieval_accuracy(val_h_i, val_h_j, ks)
    e2d_h = retrieval_accuracy(val_h_j, val_h_i, ks)
    for k in ks:
        print(f"  Top-{k}: Daimon->eFlesh {d2e_h[k]*100:.1f}% | eFlesh->Daimon {e2d_h[k]*100:.1f}%")

    print("\n===== Retrieval Accuracy (Train, z space) =====")
    d2e_train = retrieval_accuracy(train_z_i, train_z_j, ks)
    e2d_train = retrieval_accuracy(train_z_j, train_z_i, ks)
    for k in ks:
        print(f"  Top-{k}: Daimon->eFlesh {d2e_train[k]*100:.1f}% | eFlesh->Daimon {e2d_train[k]*100:.1f}%")

    # --- Summary ---
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  projection_dim         : {model_details['projection_dim']}")
    print(f"  Val  Retrieval@1 (z)   : D->E {d2e_z[1]*100:.1f}%, E->D {e2d_z[1]*100:.1f}%")
    print(f"  Train Retrieval@1 (z)  : D->E {d2e_train[1]*100:.1f}%, E->D {e2d_train[1]*100:.1f}%")
    print(f"  Cosine gap (z)         : {pos_z - neg_z:.4f}")
    print(f"  Cosine gap (h)         : {pos_h - neg_h:.4f}")
    print(f"  Plots saved to         : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
