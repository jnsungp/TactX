"""
Standalone launcher for paired-static + unpaired-dynamic alignment training
(MMD bridge + latent cycle on dynamic). Does not modify scripts/train_model.py.

Usage (defaults: ball-only static + tri_dynamic, cross_modal_vae3):
    python scripts/train_dynamic_align.py \
        --device cuda:0 \
        --run_name dynalign_balls

Resume from a Stage-1 dynamic_pretrain checkpoint:
    python scripts/train_dynamic_align.py \
        --device cuda:0 \
        --run_name dynalign_balls_from_dynpre \
        --resume_from checkpoints/cross_modal_tri_dynamic/cross_modal_vae3/<...>/model.pt

Override a new-loss hyperparameter on the fly:
    python scripts/train_dynamic_align.py \
        --device cuda:0 --run_name tune_g03_d05 \
        --gamma_mmd_max 0.3 --delta_cycle_max 0.5 --cycle_warmup_start 15
"""

import argparse
import os
from datetime import datetime

import wandb
import yaml

from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.training_dynamic_align import train_tri_dynamic_align


# Static paired objects to use for this scheme. The dynamic teleop data
# is collected on cylinder, so static is restricted to cylinder to keep
# the bridge object-consistent (ball / plug deliberately excluded).
# Object names follow the latent_data_processed_object/{pair}/{sensor}/{object}/
# layout (ball / cylinder / plug).
DEFAULT_STATIC_OBJECTS = ['cylinder']


def main():
    parser = argparse.ArgumentParser(
        description='Train cross_modal_vae3 with paired-static + dynamic-align losses.'
    )
    parser.add_argument('--dataset_name', type=str,
                        default='dataset_cross_modal_tri_object',
                        help='Paired tri-modal dataset name (de/ef/fd).')
    parser.add_argument('--dynamic_dataset_name', type=str,
                        default='dataset_cross_modal_tri_dynamic',
                        help='Unpaired per-modality dynamic dataset name (D/E/F).')
    parser.add_argument('--model_name', type=str, default='cross_modal_vae3')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--run_name', type=str, default='dynalign')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--objects', nargs='+', default=DEFAULT_STATIC_OBJECTS,
                        help='Static paired objects from latent_data_processed_object '
                             '(default: ball). Dynamic data uses all of its own '
                             'data regardless of this flag.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Optional path to a checkpoint (e.g. Stage 1 dynamic_pretrain).')

    # New-loss hyperparameter overrides (any not set fall back to defaults in
    # training_dynamic_align._gamma_mmd / _delta_cycle or to model config).
    parser.add_argument('--gamma_mmd_max', type=float, default=None)
    parser.add_argument('--mmd_warmup_epochs', type=int, default=None)
    parser.add_argument('--delta_cycle_max', type=float, default=None)
    parser.add_argument('--cycle_warmup_start', type=int, default=None)
    parser.add_argument('--cycle_warmup_epochs', type=int, default=None)
    parser.add_argument('--cycle_target', choices=['mu', 'z'], default=None)

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(repo_root, 'config', 'datasets_config.yaml')) as f:
        ds_cfg = yaml.safe_load(f)
    dataset_details = ds_cfg[args.dataset_name]

    with open(os.path.join(repo_root, 'config', 'models_config.yaml')) as f:
        mdl_cfg = yaml.safe_load(f)
    model_details = dict(mdl_cfg[args.model_name])

    # Inject new-loss hyperparameter overrides into model_details (which the
    # training fn reads via setattr → getattr(args, ...)).
    for key in ('gamma_mmd_max', 'mmd_warmup_epochs',
                'delta_cycle_max', 'cycle_warmup_start', 'cycle_warmup_epochs',
                'cycle_target'):
        v = getattr(args, key)
        if v is not None:
            model_details[key] = v

    project = 'CMTEJ_debug' if args.debug else 'CMTEJ_dynalign'
    output_folder = (f"{args.model_name}_{dataset_details['dataset_type']}_"
                     f"dynalign_{args.dataset_name}_{args.run_name}")
    wandb.init(project=project, name=output_folder, id=output_folder, resume=False)

    print(f"Run name: {args.run_name}")
    print(f"Static paired objects: {args.objects}")
    print(f"Dynamic dataset: {args.dynamic_dataset_name}")
    print(f"Checkpoint dir: checkpoints/{dataset_details['dataset_type']}/"
          f"{args.model_name}/{args.dataset_name}_run_{args.run_name}/")

    paired_train, paired_val, _ = get_all_contrastive_datasets(
        args.dataset_name, device=args.device, objects=args.objects,
    )
    dynamic_train, _dyn_val, _ = get_all_contrastive_datasets(
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
    )


if __name__ == '__main__':
    main()
