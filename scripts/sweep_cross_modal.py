"""Hyperparameter sweep runner for the cross_modal contrastive model.

Sequentially runs every combination of (learning_rate, temperature, batch_size,
projection_dim) on a single GPU, logging embedding-quality metrics to wandb so the
user can compare runs visually. No best-run auto-selection; just runs them all.

Grid (3 x 6 x 3 x 3 = 162 combinations):
    learning_rate  : [1e-4, 3e-4, 5e-4]
    temperature    : [0.03, 0.05, 0.07, 0.1, 0.2, 0.5]
    batch_size     : [64, 128, 256]
    projection_dim : [16, 32, 64]

Each run reuses the train() function from joint_embedding_learning.training with the
matching cross_modal config block from config/models_config.yaml, overriding only the
four swept fields in-memory (the yaml file is left untouched).
"""

import argparse
import copy
import gc
import itertools
import os
import sys
import traceback
from datetime import datetime

import torch
import wandb
import yaml

from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.training import train


LR_GRID          = [1e-4, 3e-4, 5e-4]
TEMPERATURE_GRID = [0.03, 0.05, 0.07, 0.1, 0.2, 0.5]
BATCH_GRID       = [64, 128, 256]
PROJ_DIM_GRID    = [16, 32, 64]


def fmt_lr(lr):
    return f"{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')


def fmt_tau(t):
    return f"{t:g}"


def build_run_name(lr, temp, bs, pd, sweep_id):
    return f"{sweep_id}_lr{fmt_lr(lr)}_tau{fmt_tau(temp)}_bs{bs}_pd{pd}"


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_configs(model_name, dataset_name):
    root = repo_root()
    with open(os.path.join(root, 'config', 'models_config.yaml'), 'r') as f:
        model_details = yaml.safe_load(f)[model_name]
    with open(os.path.join(root, 'config', 'datasets_config.yaml'), 'r') as f:
        dataset_details = yaml.safe_load(f)[dataset_name]
    return model_details, dataset_details


def checkpoint_already_done(dataset_name, dataset_type, model_name, run_name):
    ckpt = os.path.join(
        repo_root(), 'checkpoints', dataset_type, model_name,
        f"{dataset_name}_run_{run_name}", "model.pt"
    )
    return os.path.exists(ckpt)


def run_one(model_name, dataset_name, dataset_details, base_model_details,
            lr, temp, bs, pd, run_name, project, sweep_id, device, debug,
            train_datasets, validation_datasets):
    """Run a single hyperparameter configuration end-to-end."""
    model_details = copy.deepcopy(base_model_details)
    model_details['learning_rate']  = float(lr)
    model_details['temperature']    = float(temp)
    model_details['batch_size']     = int(bs)
    model_details['projection_dim'] = int(pd)

    output_folder = f"{model_name}_{dataset_details['dataset_type']}_{dataset_name}_{run_name}"

    wandb.init(
        project=project,
        name=output_folder,
        id=output_folder,
        group=sweep_id,
        config={
            'learning_rate' : lr,
            'temperature'   : temp,
            'batch_size'    : bs,
            'projection_dim': pd,
            'sweep_id'      : sweep_id,
        },
        reinit=True,
        resume=False,
    )
    try:
        train(
            train_datasets, validation_datasets,
            dataset_name, dataset_details,
            model_name, model_details,
            wandb, run_name=run_name, device=device,
        )
    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Cross-modal SimCLR hyperparameter sweep')
    parser.add_argument('--dataset_name', type=str, default='dataset_cross_modal')
    parser.add_argument('--model_name',   type=str, default='cross_modal')
    parser.add_argument('--device',       type=str, default='cuda:0')
    parser.add_argument('--project',      type=str, default='CMTEJ_sweep')
    parser.add_argument('--sweep_id',     type=str, default='',
                        help='Group tag in wandb. Defaults to a timestamp.')
    parser.add_argument('--dry_run',      action='store_true',
                        help='Print all combinations without launching training.')
    parser.add_argument('--no_skip_existing', action='store_true',
                        help='Re-run combinations even if a checkpoint exists.')
    parser.add_argument('--debug',        action='store_true')
    args = parser.parse_args()

    if not args.sweep_id:
        args.sweep_id = datetime.now().strftime('sweep_%Y%m%d_%H%M%S')

    combos = list(itertools.product(LR_GRID, TEMPERATURE_GRID, BATCH_GRID, PROJ_DIM_GRID))
    total = len(combos)
    print(f"[sweep] sweep_id={args.sweep_id}  device={args.device}")
    print(f"[sweep] total combinations: {total}")
    print(f"[sweep] grid: lr={LR_GRID}  tau={TEMPERATURE_GRID}  bs={BATCH_GRID}  pd={PROJ_DIM_GRID}")

    if args.dry_run:
        for i, (lr, t, bs, pd) in enumerate(combos):
            print(f"  [{i+1:03d}/{total}] {build_run_name(lr, t, bs, pd, args.sweep_id)}")
        return

    base_model_details, dataset_details = load_configs(args.model_name, args.dataset_name)
    dataset_type = dataset_details['dataset_type']

    print(f"[sweep] loading datasets ({args.dataset_name}) once and reusing across runs ...")
    train_datasets, val_unseen_grasps, val_unseen_tools = get_all_contrastive_datasets(
        args.dataset_name, device=args.device,
    )
    validation_datasets = {
        'val_unseen_grasps': val_unseen_grasps,
        'val_unseen_tools' : val_unseen_tools,
    }

    n_done = 0
    n_skipped = 0
    n_failed = 0
    failed_runs = []

    for idx, (lr, temp, bs, pd) in enumerate(combos, start=1):
        run_name = build_run_name(lr, temp, bs, pd, args.sweep_id)
        print(f"\n[sweep] ({idx}/{total}) lr={lr}  tau={temp}  bs={bs}  pd={pd}  ->  {run_name}")

        if (not args.no_skip_existing
                and checkpoint_already_done(args.dataset_name, dataset_type,
                                            args.model_name, run_name)):
            print(f"[sweep]   skip (checkpoint already exists)")
            n_skipped += 1
            continue

        try:
            run_one(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                dataset_details=dataset_details,
                base_model_details=base_model_details,
                lr=lr, temp=temp, bs=bs, pd=pd,
                run_name=run_name,
                project=args.project,
                sweep_id=args.sweep_id,
                device=args.device,
                debug=args.debug,
                train_datasets=train_datasets,
                validation_datasets=validation_datasets,
            )
            n_done += 1
            print(f"[sweep]   done ({n_done} total)")
        except torch.cuda.OutOfMemoryError as e:
            n_failed += 1
            failed_runs.append((run_name, 'OOM', str(e)))
            print(f"[sweep]   OOM, skipping. ({e})")
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
        except KeyboardInterrupt:
            print("[sweep] interrupted by user. Stopping.")
            try:
                wandb.finish(exit_code=255)
            except Exception:
                pass
            sys.exit(130)
        except Exception as e:
            n_failed += 1
            failed_runs.append((run_name, type(e).__name__, str(e)))
            print(f"[sweep]   FAILED with {type(e).__name__}: {e}")
            traceback.print_exc()
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n[sweep] ===== summary =====")
    print(f"[sweep] done   : {n_done}")
    print(f"[sweep] skipped: {n_skipped}")
    print(f"[sweep] failed : {n_failed}")
    if failed_runs:
        print("[sweep] failed runs:")
        for name, kind, msg in failed_runs:
            print(f"  - {name}  ({kind})  {msg[:120]}")
    print(f"[sweep] sweep_id={args.sweep_id}  project={args.project}")


if __name__ == '__main__':
    main()
