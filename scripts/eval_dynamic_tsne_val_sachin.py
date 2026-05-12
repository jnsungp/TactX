"""
Strict-val t-SNE of dynamic μ across D/E/F. Same plot as the
`tsne_dynamic.png` produced by eval_cross_modal_vae3_sachin.py, but
restricted to the **dynamic validation split** (80/20 random_split,
generator seed 0 — same convention as the rest of the codebase).

Imports the helpers from eval_cross_modal_vae3_sachin so the t-SNE setup
is identical to the main eval; this just swaps train → val for dynamic.

Usage:
    python scripts/eval_dynamic_tsne_val_sachin.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_dynalign_sachin_cylinder_<TS> \
        --device cuda:0
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.getcwd())
from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.utils.models_setup import get_model
from scripts.eval_cross_modal_vae3_sachin import (
    _SENSORS,
    extract_self_mu,
    plot_tsne_dynamic,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True,
                        help='checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_...')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dynamic_dataset_name', default='dataset_cross_modal_tri_dynamic')
    parser.add_argument('--out_dir', default=None,
                        help='Default: <checkpoint_dir>/eval_sachin/')
    parser.add_argument('--max_tsne_points', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'eval_sachin')
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model
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

    # Dynamic — TAKE THE VAL SPLIT (index [1] from get_all_contrastive_datasets).
    _dyn_train, dyn_val, _ = get_all_contrastive_datasets(
        args.dynamic_dataset_name, device='cpu',
    )
    print('Dynamic VAL sizes:', {k: len(v) for k, v in dyn_val.items()})

    # Extract μ on val
    dyn_mus = {s: extract_self_mu(dyn_val[s], s, model, args.device) for s in _SENSORS}

    # Plot — reuse the same helper, just point it at val and a new file name
    out_path = os.path.join(out_dir, 'tsne_dynamic_val.png')
    plot_tsne_dynamic(dyn_mus, out_path, n_max=args.max_tsne_points)

    print(f"\nDone. Saved: {out_path}")


if __name__ == '__main__':
    main()
