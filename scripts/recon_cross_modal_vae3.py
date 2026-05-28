"""
Self- and cross-modal reconstructions from the VAL split for a
cross_modal_vae3 checkpoint (e.g. 0TACTX).

Reuses the viz helpers in eval_cross_modal_vae3_new.py but skips the heavy
t-SNE/MMD/probe passes — only the recon panels are produced. Each panel shows
GT / Self / Cross for one sensor:

    recon_daimon.png   GT Daimon | Self D(z_D) | Cross D(z_E)
    recon_eflesh.png   GT eFlesh | Self E(z_E) | Cross E(z_D)
    recon_flexi.png    GT Flexi  | Self F(z_F) | Cross F(z_E)

Self = decode a sensor's own latent; Cross = decode the *paired* sensor's
latent through this sensor's decoder.

Usage (0TACTX is the 10-object `dataset_new`):
    python scripts/recon_cross_modal_vae3.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/0TACTX
"""

import argparse
import os
import sys

import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for eval_* import

from joint_embedding_learning.utils.models_setup import get_model
from joint_embedding_learning.data.datasets_sachin import get_paired_tri_datasets_sachin
from eval_cross_modal_vae3_new import (
    viz_recon_daimon,
    viz_recon_eflesh,
    viz_recon_flexi,
)

# Objects baked into 0TACTX (dataset_new). small_ball alone => dataset_sachin_new.
TEN_OBJECTS = [
    'circle', 'cylinder', 'cylinder_vertical', 'ellipse', 'large_ball',
    'plane', 'plane_horizontal', 'small_ball', 'square', 'triangle',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', required=True,
                    help='e.g. checkpoints/cross_modal_tri/cross_modal_vae3/0TACTX')
    ap.add_argument('--checkpoint_file', default='model.pt')
    ap.add_argument('--dataset_name', default='dataset_new',
                    help="Entry in config/datasets_config_sachin.yaml. "
                         "0TACTX -> dataset_new (10 objects).")
    ap.add_argument('--objects', nargs='+', default=TEN_OBJECTS)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n', type=int, default=8, help='examples per panel')
    ap.add_argument('--out_dir', default=None,
                    help='default: <checkpoint_dir>/recon_val')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'recon_val')
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model ──
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model('cross_modal_vae3', model_details).to(args.device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_file),
                      map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded {args.checkpoint_dir}/{args.checkpoint_file}")

    # ── Paired val split (same split logic as training) ──
    _, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device='cpu', objects=args.objects,
    )
    print('Paired val sizes:', {k: len(v) for k, v in paired_val.items()})

    # ── Recon panels (self + cross) ──
    viz_recon_daimon(paired_val['de'], model, args.device,
                     os.path.join(out_dir, 'recon_daimon.png'), n=args.n)
    viz_recon_eflesh(paired_val['de'], model, args.device,
                     os.path.join(out_dir, 'recon_eflesh.png'), n=args.n)
    viz_recon_flexi(paired_val['ef'], model, args.device,
                    os.path.join(out_dir, 'recon_flexi.png'), n=args.n)

    print(f"\nDone. Self+cross recon panels written to {out_dir}/")


if __name__ == '__main__':
    main()
