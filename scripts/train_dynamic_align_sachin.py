"""
Sachin paired-static training + L_slow + L_anchor on dynamic data.

Extends train_static_sachin.py with two temporal losses that use the
unpaired per-modality dynamic dataset (teleop episodes):

  L_slow  : ||mu_t - mu_{t+1}||^2  — consecutive-frame smoothness
  L_anchor: ||mu_nocontact||^2      — no-contact/padding frames → z_0 = 0

Both losses are zero by default so this is identical to train_static_sachin
when lambda_slow=0 and lambda_anchor=0.

Usage:
    python scripts/train_dynamic_align_sachin.py \
        --device cuda:0 \
        --run_name slow_anchor \
        --objects cylinder plug \
        --lambda_slow 0.1 \
        --lambda_anchor 0.05
"""

import argparse
import os
from datetime import datetime

import wandb
import yaml

from joint_embedding_learning.data.datasets import (
    get_all_contrastive_datasets,
    get_temporal_pair_datasets,
)
from joint_embedding_learning.data.datasets_sachin import (
    get_paired_tri_datasets_sachin,
)
from joint_embedding_learning.training_dynamic_align import train_tri_dynamic_align


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
        description='Train cross_modal_vae3 with paired-static (Sachin per-pair '
                    'rot180) + dynamic-align losses.'
    )
    parser.add_argument('--dataset_name', type=str, default='dataset_sachin',
                        help='Paired tri-modal dataset name (from '
                             'datasets_config_sachin.yaml).')
    parser.add_argument('--dynamic_dataset_name', type=str,
                        default='dataset_cross_modal_tri_dynamic',
                        help='Unpaired per-modality dynamic dataset name '
                             '(from the upstream datasets_config.yaml).')
    parser.add_argument('--model_name', type=str, default='cross_modal_vae3')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--run_name', type=str, default='dynalign_sachin')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--objects', nargs='+', default=DEFAULT_STATIC_OBJECTS,
                        help='Static paired objects (default: cylinder). '
                             'Dynamic data uses all of its own data regardless.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Optional path to a checkpoint.')

    # New-loss hyperparameter overrides.
    parser.add_argument('--gamma_mmd_max', type=float, default=None)
    parser.add_argument('--mmd_warmup_epochs', type=int, default=None)
    parser.add_argument('--delta_cycle_max', type=float, default=None)
    parser.add_argument('--cycle_warmup_start', type=int, default=None)
    parser.add_argument('--cycle_warmup_epochs', type=int, default=None)
    parser.add_argument('--cycle_target', choices=['mu', 'z'], default=None)
    parser.add_argument('--lambda_slow', type=float, default=None,
                        help='Weight for L_slow = ||mu_t - mu_{t+1}||^2 (default: 0, disabled).')
    parser.add_argument('--lambda_anchor', type=float, default=None,
                        help='Weight for L_anchor = ||mu_nocontact||^2 (default: 0, disabled).')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp

    # Paired dataset details from sachin yaml (with per-pair extra_transforms).
    dataset_details = _load_dataset_details_sachin(args.dataset_name)

    # Model config from the upstream models_config.yaml (unchanged).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, 'config', 'models_config.yaml')) as f:
        model_details = dict(yaml.safe_load(f)[args.model_name])

    # Inject new-loss hyperparameter overrides into model_details.
    for key in ('gamma_mmd_max', 'mmd_warmup_epochs',
                'delta_cycle_max', 'cycle_warmup_start',
                'cycle_warmup_epochs', 'cycle_target',
                'lambda_slow', 'lambda_anchor'):
        v = getattr(args, key)
        if v is not None:
            model_details[key] = v

    project = 'CMTEJ_debug' if args.debug else 'CMTEJ_dynalign_sachin'
    output_folder = (f"{args.model_name}_{dataset_details['dataset_type']}_"
                     f"sachin_{args.dataset_name}_{args.run_name}")
    wandb.init(project=project, name=output_folder, id=output_folder, resume=False)

    print(f"Run name: {args.run_name}")
    print(f"Paired dataset (sachin): {args.dataset_name}  objects={args.objects}")
    print(f"Dynamic dataset: {args.dynamic_dataset_name}")
    print(f"Checkpoint dir: checkpoints/{dataset_details['dataset_type']}/"
          f"{args.model_name}/{args.dataset_name}_run_{args.run_name}/")

    # Paired (sachin) with per-pair geometric corrections.
    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device=args.device, objects=args.objects,
    )

    # Dynamic via the upstream loader (no per-pair transforms needed).
    dynamic_train, _dyn_val, _ = get_all_contrastive_datasets(
        args.dynamic_dataset_name, device=args.device,
    )

    # Temporal consecutive-pair datasets for L_slow (only built if lambda_slow > 0).
    use_slow = model_details.get('lambda_slow', 0.0) or 0.0
    temporal_train = None
    if float(use_slow) > 0:
        print(f"Building temporal pair datasets from '{args.dynamic_dataset_name}' for L_slow...")
        temporal_train = get_temporal_pair_datasets(
            args.dynamic_dataset_name, device=args.device,
        )

    train_tri_dynamic_align(
        paired_train_datasets=paired_train,
        dynamic_train_datasets=dynamic_train,
        validation_datasets={'val': paired_val},
        dataset_name=args.dataset_name,
        dataset_details=dataset_details,
        model_name=args.model_name,
        model_details=model_details,
        wandb=wandb,
        run_name=args.run_name,
        device=args.device,
        resume_from=args.resume_from,
        temporal_train_datasets=temporal_train,
    )


if __name__ == '__main__':
    main()
