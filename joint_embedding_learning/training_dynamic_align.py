"""
Paired-static training with temporal smoothness and no-contact anchor losses.

Total loss:
    L = L_paired_static              (recon + InfoNCE + KL, all curricula intact)
      + λ_slow  · Σ_M  ||μ_t − μ_{t+1}||²   (consecutive dynamic frames)
      + λ_anchor· Σ_M  ||μ_nocontact||²       (no-contact/padding frames → z_0=0)
"""

import argparse
import itertools
import random

import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import make_grid

from joint_embedding_learning.training import (
    _kl_beta,
    _infonce_alpha,
    _tri_log_metrics,
    tri_validation_step,
)
from joint_embedding_learning.utils.utils import load_optimizer, save_model, get_loss
from joint_embedding_learning.utils.models_setup import get_model


_SENSORS = ('D', 'E', 'F')


def paired_step(args, paired_loaders, model, criterion, optimizer, wandb,
                dynamic_loaders=None, temporal_loaders=None):
    """One training epoch: paired static + optional L_slow + L_anchor.

    Args:
        dynamic_loaders: dict {'D': loader, ...} with is_nocontact labels.
                         Required only when lambda_anchor > 0.
        temporal_loaders: dict {'D': TemporalPairDataset loader, ...}.
                          Required only when lambda_slow > 0.
    """
    loss_epoch = 0.0
    count = 0
    lambda_slow = float(getattr(args, 'lambda_slow', 0.0))
    lambda_anchor = float(getattr(args, 'lambda_anchor', 0.0))

    max_len = max(len(l) for l in paired_loaders.values())
    last_targets = None
    last_outs = None

    # Paired: cycle shorter loaders so epoch length == longest paired loader.
    p_iters = {k: (iter(itertools.cycle(l)) if len(l) < max_len else iter(l))
               for k, l in paired_loaders.items()}

    # Dynamic: cycled independently (only needed for L_anchor).
    dyn_iters = None
    if dynamic_loaders is not None and lambda_anchor > 0:
        dyn_iters = {s: iter(itertools.cycle(dynamic_loaders[s])) for s in _SENSORS}

    # Temporal: cycled independently (only needed for L_slow).
    temp_iters = None
    if temporal_loaders is not None and lambda_slow > 0:
        temp_iters = {s: iter(itertools.cycle(temporal_loaders[s])) for s in _SENSORS}

    for batch_idx in tqdm(range(max_len)):
        paired_batches = {k: next(it) for k, it in p_iters.items()}

        targets = {pk: (x_a.to(args.device), x_b.to(args.device))
                   for pk, ((x_a, x_b), _) in paired_batches.items()}

        optimizer.zero_grad()
        outs = {p: model.forward_pair(targets[p][0], targets[p][1], p)
                for p in ('de', 'ef', 'fd')}
        beta = _kl_beta(args)
        alpha = _infonce_alpha(args)
        info = criterion(outs, targets, beta=beta, alpha=alpha)
        L_paired = info['total']

        # L_slow: ||mu_t - mu_{t+1}||^2 on consecutive dynamic frames.
        L_slow_per = {}
        L_slow_total = L_paired.new_zeros(())
        if temp_iters is not None:
            for s in _SENSORS:
                x_t, x_t1 = next(temp_iters[s])
                _, mu_t, _, _ = model._enc(x_t.to(args.device), s)
                _, mu_t1, _, _ = model._enc(x_t1.to(args.device), s)
                L_slow_per[s] = (mu_t - mu_t1).pow(2).mean()
                L_slow_total = L_slow_total + L_slow_per[s]

        # L_anchor: ||mu_nocontact||^2, no-contact frames → z_0 = 0.
        L_anchor_per = {}
        L_anchor_total = L_paired.new_zeros(())
        if dyn_iters is not None:
            for s in _SENSORS:
                batch = next(dyn_iters[s])
                x_dyn = batch[0].to(args.device)
                is_nc = batch[1][2].bool().to(args.device)
                if is_nc.sum() > 1:
                    _, mu_nc, _, _ = model._enc(x_dyn[is_nc], s)
                    L_anchor_per[s] = mu_nc.pow(2).mean()
                    L_anchor_total = L_anchor_total + L_anchor_per[s]

        loss = L_paired + lambda_slow * L_slow_total + lambda_anchor * L_anchor_total
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            log_dict = {
                "Loss_train/Model_Loss_batch": loss.item(),
                "Loss_train/paired_total": L_paired.item(),
                "Loss_train/infonce_total": info['infonce'].item(),
                "Loss_train/recon_total": info['recon'].item(),
                "Loss_train/kl_total": info['kl'].item() if torch.is_tensor(info['kl']) else float(info['kl']),
                "Loss_train/slow_total": float(L_slow_total.detach()),
                "Loss_train/anchor_total": float(L_anchor_total.detach()),
                "Loss_train/lambda_slow": lambda_slow,
                "Loss_train/lambda_anchor": lambda_anchor,
                "Loss_train/beta": float(info['beta']),
                "Loss_train/alpha_infonce": float(info['alpha_infonce']),
                "Epoch_batch": args.current_epoch,
            }
            for p in ('de', 'ef', 'fd'):
                log_dict[f"Loss_train/infonce/{p}"] = info[f"infonce_{p}"].item()
                log_dict[f"Loss_train/recon/{p}"] = info[f"recon_{p}"].item()
            for k in info:
                if k.startswith("kl_"):
                    log_dict[f"Loss_train/{k}"] = info[k].item() if torch.is_tensor(info[k]) else float(info[k])
            for s in _SENSORS:
                if s in L_slow_per:
                    log_dict[f"Loss_train/slow/{s}"] = float(L_slow_per[s].detach())
                if s in L_anchor_per:
                    log_dict[f"Loss_train/anchor/{s}"] = float(L_anchor_per[s].detach())
            log_dict.update(_tri_log_metrics("Metric_train", outs, info))

            if batch_idx % 50 == 0:
                print(
                    f"Step [{batch_idx}/{max_len}]\t"
                    f"Loss: {loss.item():.4f} "
                    f"Paired: {L_paired.item():.4f} "
                    f"Slow: {float(L_slow_total.detach()):.4f} "
                    f"Anch: {float(L_anchor_total.detach()):.4f} "
                    f"λs={lambda_slow:.3f} λa={lambda_anchor:.3f}"
                )

            wandb.log(log_dict)
            args.global_step += 1
            loss_epoch += loss.item()
            count += 1

        last_targets = targets
        last_outs = outs

    return loss_epoch / max(count, 1), last_targets, last_outs


# Keep old name as alias so any external callers don't break.
paired_with_dynamic_align_step = paired_step


def train_tri_dynamic_align(paired_train_datasets, dynamic_train_datasets,
                            validation_datasets, dataset_name, dataset_details,
                            model_name, model_details, wandb,
                            run_name='', device='cuda:0', resume_from=None,
                            temporal_train_datasets=None):
    """Joint paired-static + unpaired-dynamic-align training.

    Args:
        paired_train_datasets:  dict {'de': ds, 'ef': ds, 'fd': ds} (static).
        dynamic_train_datasets: dict {'D': ds,  'E': ds,  'F': ds} (unpaired).
        validation_datasets:    dict {split_name: {'de':..., 'ef':..., 'fd':...}}.
        resume_from:            optional path to a checkpoint (e.g. Stage 1
                                dynamic_pretrain) to load weights from.

    Extra hyperparameters (read from `model_details` via getattr, with defaults):
        lambda_slow           (0.0)      — weight for L_slow
        lambda_anchor         (0.0)      — weight for L_anchor
    """
    model_parser = argparse.ArgumentParser(description="Model-Configuration")
    model_parser.add_argument('--dataset_name', type=str, default=dataset_name)
    model_parser.add_argument('--model_name', type=str, default=model_name)
    model_parser.add_argument("--device", default=device, type=str)
    model_parser.add_argument("--run_name", type=str, default=run_name)
    model_parser.add_argument("--debug", action='store_true')
    args = model_parser.parse_args([])
    for key, value in model_details.items():
        setattr(args, key, value)
    args.resume_from = resume_from

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_type = dataset_details['dataset_type']
    model_type = model_name

    lambda_slow = float(getattr(args, 'lambda_slow', 0.0))
    lambda_anchor = float(getattr(args, 'lambda_anchor', 0.0))

    print('=' * 60)
    print('CrossModalVAE3 — paired static + L_slow + L_anchor')
    print('=' * 60)
    print(f"Batch: {args.batch_size}  LR: {args.learning_rate}")
    print(f"λ_slow={lambda_slow}  λ_anchor={lambda_anchor}")

    paired_loaders = {
        k: torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                       shuffle=True, drop_last=True)
        for k, ds in paired_train_datasets.items()
    }

    # Dynamic loaders: only built when L_anchor is active (needs is_nocontact labels).
    dynamic_loaders = None
    if lambda_anchor > 0 and dynamic_train_datasets is not None:
        dynamic_loaders = {
            k: torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                           shuffle=True, drop_last=True)
            for k, ds in dynamic_train_datasets.items()
        }
        print(f"Dynamic loaders built for L_anchor (λ={lambda_anchor})")

    # Temporal loaders: only built when L_slow is active.
    temporal_loaders = None
    if temporal_train_datasets is not None and lambda_slow > 0:
        temporal_loaders = {
            k: torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                           shuffle=True, drop_last=True)
            for k, ds in temporal_train_datasets.items()
        }
        print(f"Temporal pair loaders built for L_slow (λ={lambda_slow}): "
              f"{[f'{k}:{len(v.dataset)}' for k, v in temporal_loaders.items()]}")

    val_loaders_array = {}
    for split_key, ds_dict in validation_datasets.items():
        if not ds_dict:
            continue
        loaders = {}
        for pair_key, ds in ds_dict.items():
            if len(ds) == 0:
                continue
            loaders[pair_key] = torch.utils.data.DataLoader(
                ds, batch_size=args.batch_size, shuffle=True, drop_last=True
            )
        if len(loaders) == len(ds_dict) and len(loaders) > 0:
            val_loaders_array[split_key] = loaders

    model = get_model(model_name, model_details).to(args.device)

    if resume_from:
        state = torch.load(resume_from, map_location=args.device, weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded weights from: {resume_from}")
        if missing:
            print(f"  missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    optimizer, scheduler, _ = load_optimizer(args, model)
    criterion = get_loss(args, model_name)

    args.global_step = 0
    args.current_epoch = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epochs"):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch, last_targets, _ = paired_step(
            args, paired_loaders, model, criterion, optimizer, wandb,
            dynamic_loaders=dynamic_loaders,
            temporal_loaders=temporal_loaders,
        )
        wandb.log({"Loss_train/Model_Loss": loss_epoch,
                   "Misc/learning_rate": lr,
                   "Epoch": epoch})
        if scheduler:
            scheduler.step()

        with torch.no_grad():
            if epoch % 10 == 0:
                save_model(model, dataset_name, dataset_type, model_type,
                           dataset_details, model_details, run_name=run_name)
                if last_targets is not None:
                    x_D, _ = last_targets['de']
                    grid = make_grid(x_D[:36], nrow=6, normalize=True)
                    wandb.log({"Image_Daimon": [wandb.Image(grid)], "Epoch": epoch})

            for key, val_loaders in val_loaders_array.items():
                val_loss_epoch, _, _ = tri_validation_step(
                    args, val_loaders, model, criterion, wandb, val_name=key
                )
                if val_loss_epoch is not None:
                    wandb.log({f"Loss_{key}/Model_Loss": val_loss_epoch,
                               "Epoch": epoch})

            args.current_epoch += 1

    save_model(model, dataset_name, dataset_type, model_type,
               dataset_details, model_details, run_name=run_name)


# ---------------------------------------------------------------------------
# CLI wiring note
# ---------------------------------------------------------------------------
# To run this without modifying scripts/train_model.py, write a tiny launcher
# that loads both dataset bundles via the existing data API and calls
# train_tri_dynamic_align. Sketch:
#
#   from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
#   from joint_embedding_learning.training_dynamic_align import train_tri_dynamic_align
#
#   paired_train, paired_val, _ = get_all_contrastive_datasets(
#       'dataset_cross_modal_tri_object', device=device, objects=objects)
#   dyn_train, dyn_val, _ = get_all_contrastive_datasets(
#       'dataset_cross_modal_tri_dynamic', device=device)
#
#   train_tri_dynamic_align(
#       paired_train_datasets=paired_train,         # {de, ef, fd}
#       dynamic_train_datasets=dyn_train,           # {D, E, F}
#       validation_datasets={'val': paired_val},
#       dataset_name='dataset_cross_modal_tri_object',
#       dataset_details=...,
#       model_name='cross_modal_vae3',
#       model_details=...,           # add lambda_slow, lambda_anchor, ...
#       wandb=wandb, run_name=run_name, device=device,
#       resume_from=stage1_ckpt,     # optional dynamic_pretrain weights
#   )
