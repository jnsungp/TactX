"""
Sachin paired-static tri-modal PLAIN-CONTRASTIVE baseline.

Trains `cross_modal_clr3` (CrossModalCLR3): the three sensor encoders
(Daimon ResNet, eFlesh MLP, FlexiTac CNN) feed three plain projection heads
into one shared latent space, where 3-pair NT-Xent contrastive loss is applied
over the de / ef / fd pairs. No reconstruction, no KL — just plain contrastive
alignment, applied at full weight from epoch 0.

This is the reconstruction-free counterpart of train_static_sachin.py (which
trains cross_modal_vae3 = InfoNCE + 12-branch L1 recon + 6 KL). Same paired
dataset, same per-pair geometric corrections (rot180), same train_tri loop.

Usage:
    python scripts/train_contrastive_sachin.py \
        --device cuda:0 \
        --run_name plain_contrastive \
        --objects cylinder plug

Latent-dimension sweep (overrides model_details['projection_dim']);
run_name gets a `_pd<N>` suffix automatically so sweep runs don't collide:
    python scripts/train_contrastive_sachin.py --objects cylinder plug \
        --run_name pdsweep --projection_dim 32
"""

import argparse
import os
from datetime import datetime

import wandb
import yaml

from joint_embedding_learning.data.datasets_sachin import (
    get_paired_tri_datasets_sachin,
)
from joint_embedding_learning.training import train_tri


# Full 12-object static set in latent_data_processed_static: every object dir
# EXCEPT 'no_contact' (its thresholded twin 'no_contact_threshold' is kept).
# Matches the dataset_sachin_static12 config entry.
DEFAULT_STATIC_OBJECTS = [
    'circle', 'cylinder', 'ellipse', 'eraser', 'large_ball', 'line',
    'plug', 'small_ball', 'spam', 'triangle', 'shear', 'no_contact_threshold',
]


def _load_dataset_details_sachin(dataset_name):
    """Read dataset_details from config/datasets_config_sachin.yaml."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, 'config', 'datasets_config_sachin.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if dataset_name not in cfg:
        raise KeyError(
            f"'{dataset_name}' not in {path}; available: {list(cfg.keys())}"
        )
    return cfg[dataset_name]


def main():
    parser = argparse.ArgumentParser(
        description='Train cross_modal_clr3 (tri-modal plain-contrastive '
                    'baseline) on paired-static data. Three encoders -> shared '
                    'latent -> 3-pair NT-Xent. No recon, no KL.'
    )
    parser.add_argument('--dataset_name', type=str, default='dataset_sachin_static12',
                        help='Paired tri-modal dataset name (from '
                             'datasets_config_sachin.yaml). Default '
                             'dataset_sachin_static12 = latent_data_processed_static '
                             'with 12-object combined stats + episode val split.')
    parser.add_argument('--model_name', type=str, default='cross_modal_clr3')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--run_name', type=str, default='plain_contrastive')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--objects', nargs='+', default=DEFAULT_STATIC_OBJECTS,
                        help='Static paired objects (default: cylinder plug).')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Optional path to a checkpoint to warm-start from.')
    parser.add_argument('--projection_dim', type=int, default=None,
                        help='Override the shared latent dimension '
                             '(model_details["projection_dim"]). Used for the '
                             'latent-dim sweep. run_name gets a _pd<N> suffix.')
    parser.add_argument('--temperature', type=float, default=None,
                        help='NT-Xent temperature (overrides '
                             'model_details["temperature"], default 0.03). '
                             'run_name gets a _t<val> suffix when set.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total training epochs (overrides '
                             'model_details["epochs"], default 100).')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides '
                             'model_details["learning_rate"], default 1e-4).')
    parser.add_argument('--ef_stride', type=int, default=1,
                        help='Keep every Nth frame per episode in ef_pair. '
                             'run_name gets a _efs<N> suffix.')
    parser.add_argument('--de_stride', type=int, default=1,
                        help='Per-episode frame stride for de_pair.')
    parser.add_argument('--fd_stride', type=int, default=1,
                        help='Per-episode frame stride for fd_pair.')
    args = parser.parse_args()

    # Sweep suffixes (before timestamp) so sweep runs are distinct.
    if args.projection_dim is not None and args.run_name:
        args.run_name = f"{args.run_name}_pd{args.projection_dim}"
    if args.temperature is not None and args.run_name:
        args.run_name = f"{args.run_name}_t{args.temperature:g}"
    if args.ef_stride > 1 and args.run_name:
        args.run_name = f"{args.run_name}_efs{args.ef_stride}"
    if args.de_stride > 1 and args.run_name:
        args.run_name = f"{args.run_name}_des{args.de_stride}"
    if args.fd_stride > 1 and args.run_name:
        args.run_name = f"{args.run_name}_fds{args.fd_stride}"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp

    # Paired dataset details from sachin yaml (with per-pair extra_transforms).
    dataset_details = _load_dataset_details_sachin(args.dataset_name)

    # Model config from the upstream models_config.yaml.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, 'config', 'models_config.yaml')) as f:
        model_details = dict(yaml.safe_load(f)[args.model_name])

    if args.projection_dim is not None:
        model_details['projection_dim'] = args.projection_dim
    if args.temperature is not None:
        model_details['temperature'] = args.temperature
    if args.epochs is not None:
        model_details['epochs'] = args.epochs
    if args.lr is not None:
        model_details['learning_rate'] = args.lr

    project = 'CMTEJ_debug' if args.debug else 'CMTEJ_sachin_static'
    output_folder = (f"{args.model_name}_{dataset_details['dataset_type']}_"
                     f"sachin_{args.dataset_name}_{args.run_name}")
    wandb.init(project=project, name=output_folder, id=output_folder, resume=False)

    print(f"Run name: {args.run_name}")
    print(f"Model: {args.model_name} (tri-modal plain contrastive, no recon/KL)")
    print(f"Paired dataset (sachin): {args.dataset_name}  objects={args.objects}")
    print(f"projection_dim (latent): {model_details['projection_dim']}  "
          f"temperature: {model_details['temperature']}")
    print(f"Checkpoint dir: checkpoints/{dataset_details['dataset_type']}/"
          f"{args.model_name}/{args.dataset_name}_run_{args.run_name}/")

    # Per-pair temporal subsampling (equalize sampling rate across pairs).
    pair_strides = {k: v for k, v in
                    (('de', args.de_stride), ('ef', args.ef_stride),
                     ('fd', args.fd_stride)) if v > 1}
    if pair_strides:
        print(f"Per-pair frame strides: {pair_strides}")

    # Paired (sachin) with per-pair geometric corrections.
    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device=args.device, objects=args.objects,
        pair_strides=pair_strides or None,
    )

    train_tri(
        train_datasets=paired_train,
        validation_datasets={'val': paired_val},
        dataset_name=args.dataset_name,
        dataset_details=dataset_details,
        model_name=args.model_name,
        model_details=model_details,
        wandb=wandb,
        run_name=args.run_name,
        device=args.device,
        resume_from=args.resume_from,
    )


if __name__ == '__main__':
    main()
