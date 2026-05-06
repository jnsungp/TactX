"""Hyperparameter sweep runner — supports cross_modal and cross_modal_vae.

cross_modal (legacy grid):
    lr × temperature × batch_size × projection_dim  (3×6×3×3 = 162)

cross_modal_vae grid (the 3 most critical VAE parameters):
    lambda_recon × beta_max × temperature  (4×3×4 = 48)

    lambda_recon  : [0.1, 0.5, 1.0, 5.0]
    beta_max      : [0.001, 0.01, 0.1]
    temperature   : [0.01, 0.03, 0.07, 0.1]

Usage:
    # VAE sweep (default model=cross_modal_vae)
    python scripts/sweep_cross_modal.py \\
        --model_name cross_modal_vae \\
        --dataset_name dataset_cross_modal \\
        --device cuda:0

    # legacy cross_modal sweep
    python scripts/sweep_cross_modal.py \\
        --model_name cross_modal \\
        --dataset_name dataset_cross_modal \\
        --device cuda:0

    # dry-run (just print combos, no training)
    python scripts/sweep_cross_modal.py --model_name cross_modal_vae --dry_run
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


# ── VAE sweep grid ────────────────────────────────────────────────────────────
# 3 critical parameters identified in the previous analysis:
#   ① lambda_recon : InfoNCE ↔ reconstruction balance
#   ② beta_max     : KL regularization ceiling (posterior-collapse risk)
#   ③ temperature  : NT-Xent sharpness applied to μ
VAE_LAMBDA_RECON_GRID = [0.1, 0.5, 1.0, 3.0]
VAE_BETA_MAX_GRID     = [0.001, 0.01, 0.1]
VAE_TEMPERATURE_GRID  = [0.01, 0.03, 0.07, 0.1]

# kl_warmup_epochs is kept fixed (30) — sweep only after beta_max is chosen.
VAE_KL_WARMUP_EPOCHS  = 30

# ── Legacy cross_modal grid ───────────────────────────────────────────────────
CLR_LR_GRID          = [1e-4, 3e-4, 5e-4]
CLR_TEMPERATURE_GRID = [0.03, 0.05, 0.07, 0.1, 0.2, 0.5]
CLR_BATCH_GRID       = [64, 128, 256]
CLR_PROJ_DIM_GRID    = [16, 32, 64]


# ── Helpers ───────────────────────────────────────────────────────────────────

def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_configs(model_name, dataset_name):
    root = repo_root()
    with open(os.path.join(root, 'config', 'models_config.yaml')) as f:
        model_details = yaml.safe_load(f)[model_name]
    with open(os.path.join(root, 'config', 'datasets_config.yaml')) as f:
        dataset_details = yaml.safe_load(f)[dataset_name]
    return model_details, dataset_details


def checkpoint_done(dataset_name, dataset_type, model_name, run_name):
    ckpt = os.path.join(
        repo_root(), 'checkpoints', dataset_type, model_name,
        f"{dataset_name}_run_{run_name}", "model.pt",
    )
    return os.path.exists(ckpt)


def fmt_lr(lr):
    return f"{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')


def fmt_val(v):
    """Short string for any float hyperparameter."""
    if v == 0:
        return "0"
    if v >= 1:
        return f"{v:g}"
    # e.g. 0.001 → 1e-3, 0.5 → 0.5
    s = f"{v:.10g}"
    if len(s) > 6:
        s = f"{v:.2e}".replace('e-0', 'e-').replace('e+0', 'e+')
    return s


# ══════════════════════════════════════════════════════════════════════════════
# VAE sweep
# ══════════════════════════════════════════════════════════════════════════════

def vae_combos():
    return list(itertools.product(
        VAE_LAMBDA_RECON_GRID,
        VAE_BETA_MAX_GRID,
        VAE_TEMPERATURE_GRID,
    ))


def vae_run_name(lam, beta, tau, sweep_id):
    return (f"{sweep_id}"
            f"_lam{fmt_val(lam)}"
            f"_beta{fmt_val(beta)}"
            f"_tau{fmt_val(tau)}")


def run_one_vae(model_name, dataset_name, dataset_details, base_model_details,
                lam, beta, tau, run_name, project, sweep_id, device,
                train_datasets, validation_datasets):
    model_details = copy.deepcopy(base_model_details)
    model_details['lambda_recon']      = float(lam)
    model_details['beta_max']          = float(beta)
    model_details['temperature']       = float(tau)
    model_details['kl_warmup_epochs']  = VAE_KL_WARMUP_EPOCHS

    output_folder = f"{model_name}_{dataset_details['dataset_type']}_{dataset_name}_{run_name}"

    wandb.init(
        project=project,
        name=output_folder,
        id=output_folder,
        group=sweep_id,
        config={
            'lambda_recon'    : lam,
            'beta_max'        : beta,
            'temperature'     : tau,
            'kl_warmup_epochs': VAE_KL_WARMUP_EPOCHS,
            'sweep_id'        : sweep_id,
            'model_name'      : model_name,
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


def sweep_vae(args, base_model_details, dataset_details,
              train_datasets, validation_datasets):
    combos = vae_combos()
    total = len(combos)
    dataset_type = dataset_details['dataset_type']

    print(f"[sweep] VAE grid: "
          f"lambda_recon={VAE_LAMBDA_RECON_GRID}  "
          f"beta_max={VAE_BETA_MAX_GRID}  "
          f"temperature={VAE_TEMPERATURE_GRID}")
    print(f"[sweep] total combinations: {total}  "
          f"(kl_warmup_epochs fixed at {VAE_KL_WARMUP_EPOCHS})")

    if args.dry_run:
        for i, (lam, beta, tau) in enumerate(combos):
            rn = vae_run_name(lam, beta, tau, args.sweep_id)
            print(f"  [{i+1:03d}/{total}] lam={lam}  beta={beta}  tau={tau}  → {rn}")
        return

    n_done = n_skipped = n_failed = 0
    failed_runs = []

    for idx, (lam, beta, tau) in enumerate(combos, start=1):
        run_name = vae_run_name(lam, beta, tau, args.sweep_id)
        print(f"\n[sweep] ({idx}/{total})  "
              f"lambda_recon={lam}  beta_max={beta}  tau={tau}  →  {run_name}")

        if (not args.no_skip_existing
                and checkpoint_done(args.dataset_name, dataset_type,
                                    args.model_name, run_name)):
            print("[sweep]   skip (checkpoint already exists)")
            n_skipped += 1
            continue

        try:
            run_one_vae(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                dataset_details=dataset_details,
                base_model_details=base_model_details,
                lam=lam, beta=beta, tau=tau,
                run_name=run_name,
                project=args.project,
                sweep_id=args.sweep_id,
                device=args.device,
                train_datasets=train_datasets,
                validation_datasets=validation_datasets,
            )
            n_done += 1
            print(f"[sweep]   done  ({n_done} completed so far)")
        except torch.cuda.OutOfMemoryError as e:
            n_failed += 1
            failed_runs.append((run_name, 'OOM', str(e)))
            print(f"[sweep]   OOM — skipping  ({e})")
            _safe_wandb_finish(1)
        except KeyboardInterrupt:
            print("[sweep] interrupted by user.")
            _safe_wandb_finish(255)
            sys.exit(130)
        except Exception as e:
            n_failed += 1
            failed_runs.append((run_name, type(e).__name__, str(e)))
            print(f"[sweep]   FAILED  {type(e).__name__}: {e}")
            traceback.print_exc()
            _safe_wandb_finish(1)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _print_summary(n_done, n_skipped, n_failed, failed_runs, args)


# ══════════════════════════════════════════════════════════════════════════════
# Legacy cross_modal sweep (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

def clr_run_name(lr, temp, bs, pd, sweep_id):
    return (f"{sweep_id}"
            f"_lr{fmt_lr(lr)}"
            f"_tau{fmt_val(temp)}"
            f"_bs{bs}"
            f"_pd{pd}")


def run_one_clr(model_name, dataset_name, dataset_details, base_model_details,
                lr, temp, bs, pd, run_name, project, sweep_id, device,
                train_datasets, validation_datasets):
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


def sweep_clr(args, base_model_details, dataset_details,
              train_datasets, validation_datasets):
    combos = list(itertools.product(
        CLR_LR_GRID, CLR_TEMPERATURE_GRID, CLR_BATCH_GRID, CLR_PROJ_DIM_GRID))
    total = len(combos)
    dataset_type = dataset_details['dataset_type']

    print(f"[sweep] cross_modal grid: "
          f"lr={CLR_LR_GRID}  tau={CLR_TEMPERATURE_GRID}  "
          f"bs={CLR_BATCH_GRID}  pd={CLR_PROJ_DIM_GRID}")
    print(f"[sweep] total combinations: {total}")

    if args.dry_run:
        for i, (lr, t, bs, pd) in enumerate(combos):
            rn = clr_run_name(lr, t, bs, pd, args.sweep_id)
            print(f"  [{i+1:03d}/{total}] {rn}")
        return

    n_done = n_skipped = n_failed = 0
    failed_runs = []

    for idx, (lr, temp, bs, pd) in enumerate(combos, start=1):
        run_name = clr_run_name(lr, temp, bs, pd, args.sweep_id)
        print(f"\n[sweep] ({idx}/{total})  "
              f"lr={lr}  tau={temp}  bs={bs}  pd={pd}  →  {run_name}")

        if (not args.no_skip_existing
                and checkpoint_done(args.dataset_name, dataset_type,
                                    args.model_name, run_name)):
            print("[sweep]   skip (checkpoint already exists)")
            n_skipped += 1
            continue

        try:
            run_one_clr(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                dataset_details=dataset_details,
                base_model_details=base_model_details,
                lr=lr, temp=temp, bs=bs, pd=pd,
                run_name=run_name,
                project=args.project,
                sweep_id=args.sweep_id,
                device=args.device,
                train_datasets=train_datasets,
                validation_datasets=validation_datasets,
            )
            n_done += 1
            print(f"[sweep]   done  ({n_done} completed so far)")
        except torch.cuda.OutOfMemoryError as e:
            n_failed += 1
            failed_runs.append((run_name, 'OOM', str(e)))
            print(f"[sweep]   OOM — skipping  ({e})")
            _safe_wandb_finish(1)
        except KeyboardInterrupt:
            print("[sweep] interrupted by user.")
            _safe_wandb_finish(255)
            sys.exit(130)
        except Exception as e:
            n_failed += 1
            failed_runs.append((run_name, type(e).__name__, str(e)))
            print(f"[sweep]   FAILED  {type(e).__name__}: {e}")
            traceback.print_exc()
            _safe_wandb_finish(1)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _print_summary(n_done, n_skipped, n_failed, failed_runs, args)


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def _safe_wandb_finish(exit_code=0):
    try:
        wandb.finish(exit_code=exit_code)
    except Exception:
        pass


def _print_summary(n_done, n_skipped, n_failed, failed_runs, args):
    print("\n[sweep] ═══════════════ summary ═══════════════")
    print(f"[sweep] done   : {n_done}")
    print(f"[sweep] skipped: {n_skipped}")
    print(f"[sweep] failed : {n_failed}")
    if failed_runs:
        print("[sweep] failed runs:")
        for name, kind, msg in failed_runs:
            print(f"  - {name}  ({kind})  {msg[:120]}")
    print(f"[sweep] sweep_id={args.sweep_id}  project={args.project}")
    print("[sweep] ══════════════════════════════════════")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Cross-modal hyperparameter sweep (cross_modal / cross_modal_vae)')
    parser.add_argument('--model_name',   type=str, default='cross_modal_vae',
                        choices=['cross_modal', 'cross_modal_vae'],
                        help='Model to sweep. Determines which grid is used.')
    parser.add_argument('--dataset_name', type=str, default='dataset_cross_modal')
    parser.add_argument('--device',       type=str, default='cuda:0')
    parser.add_argument('--project',      type=str, default='CMTEJ_sweep')
    parser.add_argument('--sweep_id',     type=str, default='',
                        help='Wandb group tag. Defaults to a timestamp.')
    parser.add_argument('--dry_run',      action='store_true',
                        help='Print all combinations without launching training.')
    parser.add_argument('--no_skip_existing', action='store_true',
                        help='Re-run even if a checkpoint already exists.')
    args = parser.parse_args()

    if not args.sweep_id:
        args.sweep_id = datetime.now().strftime('sweep_%Y%m%d_%H%M%S')

    print(f"[sweep] model_name={args.model_name}  sweep_id={args.sweep_id}  device={args.device}")

    base_model_details, dataset_details = load_configs(args.model_name, args.dataset_name)

    if not args.dry_run:
        print(f"[sweep] loading datasets ({args.dataset_name}) once …")
        train_datasets, val_unseen_grasps, val_unseen_tools = get_all_contrastive_datasets(
            args.dataset_name, device=args.device,
        )
        validation_datasets = {
            'val_unseen_grasps': val_unseen_grasps,
            'val_unseen_tools' : val_unseen_tools,
        }
    else:
        train_datasets = validation_datasets = None

    if args.model_name == 'cross_modal_vae':
        sweep_vae(args, base_model_details, dataset_details,
                  train_datasets, validation_datasets)
    else:
        sweep_clr(args, base_model_details, dataset_details,
                  train_datasets, validation_datasets)


if __name__ == '__main__':
    main()
