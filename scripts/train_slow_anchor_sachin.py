"""
train_static_sachin.py + L_slow + L_anchor on dynamic data.

Everything is identical to train_static_sachin.py (same paired dataset,
norm stats, strides, warmup curricula, wandb project). Two extra losses are
bolted on using the unpaired dynamic teleop data:

  L_slow  : ||mu_t - mu_{t+1}||^2  — consecutive-frame smoothness in latent space
  L_anchor: ||mu_nocontact||^2      — no-contact (padding) frames pulled to z_0 = 0

Both default to 0 so the script is a drop-in for train_static_sachin when
lambda_slow=0 and lambda_anchor=0.

Usage:
    python scripts/train_slow_anchor_sachin.py \
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


DEFAULT_STATIC_OBJECTS = [
    'circle', 'cylinder', 'ellipse', 'eraser', 'large_ball',
    'line', 'plug', 'shear', 'small_ball', 'spam', 'triangle',
]


def _load_dataset_details_sachin(dataset_name):
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
        description='Paired-static tri-modal training + L_slow + L_anchor. '
                    'Identical to train_static_sachin.py plus two optional '
                    'temporal losses on dynamic teleop data.'
    )
    # --- identical to train_static_sachin ---
    parser.add_argument('--dataset_name', type=str, default='dataset_sachin_padded')
    parser.add_argument('--model_name', type=str, default='cross_modal_vae3')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--run_name', type=str, default='slow_anchor')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--objects', nargs='+', default=DEFAULT_STATIC_OBJECTS)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--projection_dim', type=int, default=None)
    parser.add_argument('--infonce_warmup_start', type=int, default=None)
    parser.add_argument('--infonce_warmup_epochs', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--ef_stride', type=int, default=1)
    parser.add_argument('--de_stride', type=int, default=1)
    parser.add_argument('--fd_stride', type=int, default=1)
    # --- new ---
    parser.add_argument('--dynamic_dataset_name', type=str,
                        default='dataset_cross_modal_tri_dynamic_sachin_padded',
                        help='Unpaired per-modality dynamic dataset for L_slow/L_anchor.')
    parser.add_argument('--lambda_slow', type=float, default=0.0,
                        help='Weight for L_slow = ||mu_t - mu_{t+1}||^2.')
    parser.add_argument('--lambda_anchor', type=float, default=0.0,
                        help='Weight for L_anchor = ||mu_nocontact||^2.')
    args = parser.parse_args()

    # --- identical sweep suffixes from train_static_sachin ---
    if args.projection_dim is not None and args.run_name:
        args.run_name = f"{args.run_name}_pd{args.projection_dim}"
    if args.infonce_warmup_start is not None and args.run_name:
        args.run_name = f"{args.run_name}_ws{args.infonce_warmup_start}"
    if args.infonce_warmup_epochs is not None and args.run_name:
        args.run_name = f"{args.run_name}_we{args.infonce_warmup_epochs}"
    if args.ef_stride > 1 and args.run_name:
        args.run_name = f"{args.run_name}_efs{args.ef_stride}"
    if args.de_stride > 1 and args.run_name:
        args.run_name = f"{args.run_name}_des{args.de_stride}"
    if args.fd_stride > 1 and args.run_name:
        args.run_name = f"{args.run_name}_fds{args.fd_stride}"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp

    dataset_details = _load_dataset_details_sachin(args.dataset_name)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, 'config', 'models_config.yaml')) as f:
        model_details = dict(yaml.safe_load(f)[args.model_name])

    # --- identical overrides from train_static_sachin ---
    if args.projection_dim is not None:
        model_details['projection_dim'] = args.projection_dim
    if args.infonce_warmup_start is not None:
        model_details['infonce_warmup_start'] = args.infonce_warmup_start
    if args.infonce_warmup_epochs is not None:
        model_details['infonce_warmup_epochs'] = args.infonce_warmup_epochs
    if args.epochs is not None:
        model_details['epochs'] = args.epochs
    if args.lr is not None:
        model_details['learning_rate'] = args.lr

    # --- new loss weights ---
    model_details['lambda_slow'] = args.lambda_slow
    model_details['lambda_anchor'] = args.lambda_anchor

    project = 'CMTEJ_debug' if args.debug else 'CMTEJ_static_sachin'
    output_folder = (f"{args.model_name}_{dataset_details['dataset_type']}_"
                     f"sachin_{args.dataset_name}_{args.run_name}")
    wandb.init(project=project, name=output_folder, id=output_folder, resume=False)

    print(f"Run name: {args.run_name}")
    print(f"Paired dataset (sachin): {args.dataset_name}  objects={args.objects}")
    print(f"projection_dim: {model_details['projection_dim']}")
    print(f"lambda_slow={args.lambda_slow}  lambda_anchor={args.lambda_anchor}")
    print(f"Checkpoint dir: checkpoints/{dataset_details['dataset_type']}/"
          f"{args.model_name}/{args.dataset_name}_run_{args.run_name}/")

    pair_strides = {k: v for k, v in
                    (('de', args.de_stride), ('ef', args.ef_stride),
                     ('fd', args.fd_stride)) if v > 1}
    if pair_strides:
        print(f"Per-pair frame strides: {pair_strides}")

    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device=args.device, objects=args.objects,
        pair_strides=pair_strides or None,
    )

    # Dynamic data — only loaded when at least one temporal loss is active.
    dynamic_train = None
    temporal_train = None
    if args.lambda_slow > 0 or args.lambda_anchor > 0:
        print(f"Loading dynamic dataset '{args.dynamic_dataset_name}'...")
        dynamic_train, _, _ = get_all_contrastive_datasets(
            args.dynamic_dataset_name, device=args.device,
        )
        if args.lambda_slow > 0:
            print("Building temporal pair datasets for L_slow...")
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
