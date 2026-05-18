"""
Sachin paired-static tri-modal training with the 'cwcl' objective.

Same as scripts/train_static_sachin.py EXCEPT the cross-modal alignment loss:
this injects `loss: 'cwcl'` into model_details, so get_loss builds
CrossModalVAE3CWCLLoss — Continuous Weighted Contrastive Loss. Instead of
InfoNCE (only the diagonal is a positive), CWCL gives graded soft-positive
weight exp(-|Δframe|/sigma) to nearby same-(object,episode) frames.

CWCL is a strict generalization of InfoNCE (sigma→0 ⇒ CWCL == InfoNCE), so it
*replaces* the contrastive term — do not also run InfoNCE.

Episode-aware batching: CWCL's soft positives only fire when same-episode
frames share a batch. train_tri auto-enables EpisodeChunkBatchSampler when
loss == 'cwcl' (batches built from contiguous frame-chunks). Watch the
wandb `Loss_train/cwcl_offdiag_mass_*` curves — if they sit near 0 the soft
positives aren't firing and CWCL has degraded to InfoNCE.

Everything else matches train_static_sachin.py: paired-static tri-modal
training over de/ef/fd, recon (4× L1/pair) + KL, per-pair rot180, no dynamic
data / no MMD / no cycle.

Usage:
    python scripts/train_cwcl_sachin.py \
        --device cuda:0 --run_name cwcl \
        --dataset_name dataset_sachin_static \
        --objects circle cylinder ellipse eraser large_ball line plug \
                  small_ball spam triangle

CWCL hyperparameters:
    --cwcl_sigma N        frame-distance kernel width (default 5.0)
    --cwcl_chunk_size N   frames per contiguous chunk in a batch (default 8)
    --cwcl_cross_episode  weight same-object frames across episodes too
                          (default: same-episode-only)
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


DEFAULT_STATIC_OBJECTS = ['cylinder', 'plug']

ALIGNMENT_LOSS = 'cwcl'


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
        description='Train cross_modal_vae3 on paired-static data with the '
                    'CWCL (soft-weighted contrastive) alignment objective.'
    )
    parser.add_argument('--dataset_name', type=str, default='dataset_sachin')
    parser.add_argument('--model_name', type=str, default='cross_modal_vae3')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--run_name', type=str, default='cwcl')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--objects', nargs='+', default=DEFAULT_STATIC_OBJECTS)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--projection_dim', type=int, default=None)
    parser.add_argument('--infonce_warmup_start', type=int, default=None,
                        help='Epochs of recon-only before CWCL turns on.')
    parser.add_argument('--infonce_warmup_epochs', type=int, default=None,
                        help='Ramp width for the alignment weight alpha.')
    parser.add_argument('--epochs', type=int, default=None)
    # CWCL-specific.
    parser.add_argument('--cwcl_sigma', type=float, default=None,
                        help='Frame-distance kernel width (default 5.0). '
                             'Smaller -> CWCL closer to plain InfoNCE.')
    parser.add_argument('--cwcl_chunk_size', type=int, default=None,
                        help='Contiguous frames per chunk in a batch '
                             '(default 8). Drives soft-positive density.')
    parser.add_argument('--cwcl_cross_episode', action='store_true',
                        help='Allow soft positives across episodes of the '
                             'same object (default: same-episode-only).')
    args = parser.parse_args()

    # Sweep suffixes (before timestamp) so runs are distinct.
    if args.projection_dim is not None and args.run_name:
        args.run_name = f"{args.run_name}_pd{args.projection_dim}"
    if args.infonce_warmup_start is not None and args.run_name:
        args.run_name = f"{args.run_name}_ws{args.infonce_warmup_start}"
    if args.infonce_warmup_epochs is not None and args.run_name:
        args.run_name = f"{args.run_name}_we{args.infonce_warmup_epochs}"
    if args.cwcl_sigma is not None and args.run_name:
        args.run_name = f"{args.run_name}_sig{args.cwcl_sigma:g}"
    if args.cwcl_chunk_size is not None and args.run_name:
        args.run_name = f"{args.run_name}_ch{args.cwcl_chunk_size}"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp

    dataset_details = _load_dataset_details_sachin(args.dataset_name)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, 'config', 'models_config.yaml')) as f:
        model_details = dict(yaml.safe_load(f)[args.model_name])

    # Select CWCL. get_loss reads model_details['loss']; train_tri auto-enables
    # the episode-aware batch sampler when loss == 'cwcl'.
    model_details['loss'] = ALIGNMENT_LOSS

    if args.projection_dim is not None:
        model_details['projection_dim'] = args.projection_dim
    if args.infonce_warmup_start is not None:
        model_details['infonce_warmup_start'] = args.infonce_warmup_start
    if args.infonce_warmup_epochs is not None:
        model_details['infonce_warmup_epochs'] = args.infonce_warmup_epochs
    if args.epochs is not None:
        model_details['epochs'] = args.epochs
    if args.cwcl_sigma is not None:
        model_details['cwcl_sigma'] = args.cwcl_sigma
    if args.cwcl_chunk_size is not None:
        model_details['cwcl_chunk_size'] = args.cwcl_chunk_size
    model_details['cwcl_same_episode_only'] = not args.cwcl_cross_episode

    project = 'CMTEJ_debug' if args.debug else 'CMTEJ_cwcl_sachin'
    output_folder = (f"{args.model_name}_{dataset_details['dataset_type']}_"
                     f"sachin_{args.dataset_name}_{args.run_name}")
    wandb.init(project=project, name=output_folder, id=output_folder, resume=False)

    print(f"Run name: {args.run_name}")
    print(f"Alignment loss: cwcl  "
          f"(sigma={model_details.get('cwcl_sigma', 5.0)}, "
          f"chunk={model_details.get('cwcl_chunk_size', 8)}, "
          f"same_episode_only={model_details['cwcl_same_episode_only']})")
    print(f"Paired dataset (sachin): {args.dataset_name}  objects={args.objects}")
    print(f"projection_dim (latent): {model_details['projection_dim']}")
    print(f"Checkpoint dir: checkpoints/{dataset_details['dataset_type']}/"
          f"{args.model_name}/{args.dataset_name}_run_{args.run_name}/")

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
