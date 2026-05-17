"""
Sachin variant of plain paired-static tri-modal training — no dynamic-align.

This is the *true* "plain" baseline. The dynamic-align trainer
(train_dynamic_align_sachin.py) still runs the full dynamic forward
(`model.forward_self` on D/E/F) and the MMD computation every step even when
`gamma_mmd_max=0` — the zero-weighted term still builds its autograd graph and
holds dynamic-branch activations in memory until backward. That wastes RAM and
step time. This file calls `train_tri` instead, which never touches dynamic
data: paired-static tri-modal training only (recon + KL + InfoNCE over the
de / ef / fd pairs).

Paired dataset is loaded from config/datasets_config_sachin.yaml via
joint_embedding_learning.data.datasets_sachin, so per-pair geometric
corrections (rot180) are applied at __getitem__ time — identical to the
dynalign script's paired side.

Usage:
    python scripts/train_static_sachin.py \
        --device cuda:0 \
        --run_name plain_static \
        --objects cylinder plug

Latent-dimension sweep (overrides model_details['projection_dim']);
run_name gets a `_pd<N>` suffix automatically so sweep runs don't collide:
    python scripts/train_static_sachin.py --objects cylinder plug \
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


# Static paired objects. Joint cyl+plug by default (matches the Phase-3 setup).
DEFAULT_STATIC_OBJECTS = ['cylinder', 'plug']


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
        description='Train cross_modal_vae3 on paired-static data only '
                    '(Sachin per-pair rot180). No MMD, no cycle, no dynamic '
                    'data — the lean "plain" baseline.'
    )
    parser.add_argument('--dataset_name', type=str, default='dataset_sachin',
                        help='Paired tri-modal dataset name (from '
                             'datasets_config_sachin.yaml).')
    parser.add_argument('--model_name', type=str, default='cross_modal_vae3')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--run_name', type=str, default='plain_static')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--objects', nargs='+', default=DEFAULT_STATIC_OBJECTS,
                        help='Static paired objects (default: cylinder plug).')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Optional path to a checkpoint to warm-start from.')
    parser.add_argument('--projection_dim', type=int, default=None,
                        help='Override the shared latent dimension '
                             '(model_details["projection_dim"]). Used for '
                             'the latent-dim sweep. run_name gets a _pd<N> '
                             'suffix when this is set.')
    parser.add_argument('--infonce_warmup_start', type=int, default=None,
                        help='Epochs of recon-only training before the '
                             'alignment (InfoNCE) term turns on. Overrides '
                             'model_details. run_name gets a _ws<N> suffix.')
    parser.add_argument('--infonce_warmup_epochs', type=int, default=None,
                        help='Ramp width: epochs over which the alignment '
                             'weight alpha goes 0->1 after the start epoch. '
                             'Overrides model_details. run_name gets _we<N>.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total training epochs (overrides '
                             'model_details["epochs"], default 100).')
    args = parser.parse_args()

    # Sweep suffixes (before timestamp) so sweep runs are distinct.
    if args.projection_dim is not None and args.run_name:
        args.run_name = f"{args.run_name}_pd{args.projection_dim}"
    if args.infonce_warmup_start is not None and args.run_name:
        args.run_name = f"{args.run_name}_ws{args.infonce_warmup_start}"
    if args.infonce_warmup_epochs is not None and args.run_name:
        args.run_name = f"{args.run_name}_we{args.infonce_warmup_epochs}"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp

    # Paired dataset details from sachin yaml (with per-pair extra_transforms).
    dataset_details = _load_dataset_details_sachin(args.dataset_name)

    # Model config from the upstream models_config.yaml.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, 'config', 'models_config.yaml')) as f:
        model_details = dict(yaml.safe_load(f)[args.model_name])

    # Latent-dim override for sweeps. projection_dim flows to the projection
    # heads (2*pd output) and all three decoders (pd in_dim) — single knob.
    if args.projection_dim is not None:
        model_details['projection_dim'] = args.projection_dim

    # Curriculum override: delay/ramp the alignment (InfoNCE) term so the
    # latent learns reconstruction first. See _infonce_alpha in training.py.
    if args.infonce_warmup_start is not None:
        model_details['infonce_warmup_start'] = args.infonce_warmup_start
    if args.infonce_warmup_epochs is not None:
        model_details['infonce_warmup_epochs'] = args.infonce_warmup_epochs
    if args.epochs is not None:
        model_details['epochs'] = args.epochs

    project = 'CMTEJ_debug' if args.debug else 'CMTEJ_static_sachin'
    output_folder = (f"{args.model_name}_{dataset_details['dataset_type']}_"
                     f"sachin_{args.dataset_name}_{args.run_name}")
    wandb.init(project=project, name=output_folder, id=output_folder, resume=False)

    print(f"Run name: {args.run_name}")
    print(f"Paired dataset (sachin): {args.dataset_name}  objects={args.objects}")
    print(f"projection_dim (latent): {model_details['projection_dim']}")
    print(f"Checkpoint dir: checkpoints/{dataset_details['dataset_type']}/"
          f"{args.model_name}/{args.dataset_name}_run_{args.run_name}/")

    # Paired (sachin) with per-pair geometric corrections.
    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device=args.device, objects=args.objects,
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
