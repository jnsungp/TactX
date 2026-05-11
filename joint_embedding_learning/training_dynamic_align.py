"""
Paired-static training augmented with unpaired-dynamic alignment.

Adds two new losses on top of the existing tri-modal paired training
(`train_tri`) so that dynamic-distribution inputs land in the same latent
region as the cross-aligned static distribution:

  1. MMD bridge
        For each modality M in {D, E, F}, match the marginal of dynamic-M μ
        to static-M μ. Cross-modal alignment is inherited transitively from
        the static InfoNCE hub. Distributional, no pairs required.

  2. Cycle loss (on dynamic data)
        x_M --Enc_M--> z_M --Dec_N--> x̂_N --Enc_N--> (μ_N', z_N')
        Enforce μ_N' ≈ μ_M (or z_N' ≈ z_M). Per-sample cross-modal pull,
        no pairs required. Uses the cross-decoders already in the model.

Total loss:
    L = L_paired_static              (existing tri loss, all curricula intact)
      + γ · Σ_M  MMD(static-μ_M, dynamic-μ_M)
      + δ · Σ_(src,tgt)  Cycle(src→tgt)

`δ` is delay-warmed up so cycle only kicks in once cross-decoders are usable.

Drop-in usage:
    from joint_embedding_learning.training_dynamic_align import train_tri_dynamic_align
    train_tri_dynamic_align(
        paired_train_datasets={'de': ..., 'ef': ..., 'fd': ...},
        dynamic_train_datasets={'D': ..., 'E': ..., 'F': ...},
        validation_datasets={'val': {'de': ..., 'ef': ..., 'fd': ...}},
        ...
    )
"""

import argparse
import itertools
import random

import numpy as np
import torch
import torch.nn.functional as F
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

_CYCLE_DIRECTIONS = (
    ('D', 'E'), ('E', 'D'),
    ('E', 'F'), ('F', 'E'),
    ('F', 'D'), ('D', 'F'),
)

_DEFAULT_BANDWIDTHS = (1.0, 2.0, 4.0, 8.0, 16.0)


def mmd_rbf_multi(x, y, bandwidths=_DEFAULT_BANDWIDTHS):
    """Multi-kernel RBF MMD² between two unpaired batches.

    Args:
        x: (n, d) samples from P.
        y: (m, d) samples from Q.
        bandwidths: 2σ² values for the mixture of RBF kernels. Mixture is
            standard practice — robust to the unknown scale of the latent.

    Returns:
        Scalar MMD² (differentiable).
    """
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    mmd2 = x.new_zeros(())
    for s in bandwidths:
        k_xx = torch.exp(-xx / (2.0 * s))
        k_yy = torch.exp(-yy / (2.0 * s))
        k_xy = torch.exp(-xy / (2.0 * s))
        mmd2 = mmd2 + (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return mmd2 / float(len(bandwidths))


def cycle_loss_direction(model, x_src, src, tgt, target='mu'):
    """One cycle direction src→tgt→src on an unpaired sample.

    z_src = Enc_src(x_src)
    x_hat_tgt = Dec_tgt(z_src)        # cross-decoder, already in model
    (μ_tgt', _, _, z_tgt') = Enc_tgt(x_hat_tgt)

    Returns L1 between (μ_src, μ_tgt') if target='mu' (default, deterministic
    and more stable), or (z_src, z_tgt') if target='z'.
    """
    _, mu_s, _, z_s = model._enc(x_src, src)
    x_hat_t = model._decoder_for(tgt)(z_s)
    _, mu_t, _, z_t = model._enc(x_hat_t, tgt)
    if target == 'mu':
        return F.l1_loss(mu_t, mu_s)
    if target == 'z':
        return F.l1_loss(z_t, z_s)
    raise ValueError(f"Unknown cycle target '{target}', expected 'mu' or 'z'")


def _gamma_mmd(args):
    """Linear warmup of MMD weight 0 → gamma_mmd_max over mmd_warmup_epochs."""
    g_max = float(getattr(args, 'gamma_mmd_max', 0.5))
    n = int(getattr(args, 'mmd_warmup_epochs', 0))
    if n <= 0:
        return g_max
    return g_max * min(args.current_epoch / n, 1.0)


def _delta_cycle(args):
    """Delayed warmup of cycle weight.

    Stays 0 until cycle_warmup_start, then ramps 0 → delta_cycle_max over
    cycle_warmup_epochs. Delayed because cycle is meaningful only once the
    cross-decoders produce something usable.
    """
    d_max = float(getattr(args, 'delta_cycle_max', 0.5))
    start = int(getattr(args, 'cycle_warmup_start', 10))
    n = int(getattr(args, 'cycle_warmup_epochs', 10))
    e = args.current_epoch
    if e < start:
        return 0.0
    if n <= 0:
        return d_max
    return d_max * min((e - start) / n, 1.0)


def _zip_paired_and_dynamic(paired_loaders, dynamic_loaders):
    """Yield (paired_batches, dynamic_batches) per step.

    Epoch length = max of all six loader lengths; shorter loaders are cycled.
    """
    all_lens = [len(l) for l in paired_loaders.values()] + \
               [len(l) for l in dynamic_loaders.values()]
    max_len = max(all_lens)
    p_iters = {k: (iter(itertools.cycle(l)) if len(l) < max_len else iter(l))
               for k, l in paired_loaders.items()}
    d_iters = {k: (iter(itertools.cycle(l)) if len(l) < max_len else iter(l))
               for k, l in dynamic_loaders.items()}
    for _ in range(max_len):
        yield (
            {k: next(it) for k, it in p_iters.items()},
            {k: next(it) for k, it in d_iters.items()},
        )


def paired_with_dynamic_align_step(args, paired_loaders, dynamic_loaders,
                                   model, criterion, optimizer, wandb):
    """One training epoch: paired static + MMD bridge + cycle on dynamic."""
    loss_epoch = 0.0
    count = 0
    max_len = max(
        max(len(l) for l in paired_loaders.values()),
        max(len(l) for l in dynamic_loaders.values()),
    )
    last_targets = None
    last_outs = None

    gamma = _gamma_mmd(args)
    delta = _delta_cycle(args)
    bw = tuple(getattr(args, 'mmd_bandwidths', _DEFAULT_BANDWIDTHS))
    cycle_target = getattr(args, 'cycle_target', 'mu')

    iterable = _zip_paired_and_dynamic(paired_loaders, dynamic_loaders)
    for batch_idx, (paired_batches, dyn_batches) in enumerate(
            tqdm(iterable, total=max_len)):

        # Paired (static) forward, identical to tri_train_step.
        targets = {}
        for pk, ((x_a, x_b), _labels) in paired_batches.items():
            targets[pk] = (x_a.to(args.device), x_b.to(args.device))

        optimizer.zero_grad()
        outs = {p: model.forward_pair(targets[p][0], targets[p][1], p)
                for p in ('de', 'ef', 'fd')}
        beta = _kl_beta(args)
        alpha = _infonce_alpha(args)
        info = criterion(outs, targets, beta=beta, alpha=alpha)
        L_paired = info['total']

        # Dynamic (unpaired, per-modality) forward.
        dyn_inputs = {s: dyn_batches[s][0].to(args.device) for s in _SENSORS}
        dyn_outs = {s: model.forward_self(dyn_inputs[s], s) for s in _SENSORS}

        # MMD bridge: static μ vs dynamic μ, per modality.
        # Pick one canonical paired-forward source per sensor to avoid
        # double-counting (each modality appears in two paired forwards):
        #   D ← outs['de']['mu_a']   (de = D,E)
        #   E ← outs['de']['mu_b']
        #   F ← outs['ef']['mu_b']   (ef = E,F)
        static_mu = {
            'D': outs['de']['mu_a'],
            'E': outs['de']['mu_b'],
            'F': outs['ef']['mu_b'],
        }
        L_mmd_per = {}
        L_mmd_total = L_paired.new_zeros(())
        for s in _SENSORS:
            m = mmd_rbf_multi(static_mu[s], dyn_outs[s]['mu'], bandwidths=bw)
            L_mmd_per[s] = m
            L_mmd_total = L_mmd_total + m

        # Cycle on dynamic data (delayed by curriculum).
        L_cyc_per = {}
        L_cyc_total = L_paired.new_zeros(())
        if delta > 0:
            for src, tgt in _CYCLE_DIRECTIONS:
                c = cycle_loss_direction(model, dyn_inputs[src], src, tgt,
                                         target=cycle_target)
                L_cyc_per[(src, tgt)] = c
                L_cyc_total = L_cyc_total + c

        loss = L_paired + gamma * L_mmd_total + delta * L_cyc_total
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            log_dict = {
                "Loss_train/Model_Loss_batch": loss.item(),
                "Loss_train/paired_total": L_paired.item(),
                "Loss_train/infonce_total": info['infonce'].item(),
                "Loss_train/recon_total": info['recon'].item(),
                "Loss_train/kl_total": info['kl'].item() if torch.is_tensor(info['kl']) else float(info['kl']),
                "Loss_train/mmd_total": float(L_mmd_total.detach()),
                "Loss_train/cycle_total": float(L_cyc_total.detach()) if delta > 0 else 0.0,
                "Loss_train/gamma_mmd": float(gamma),
                "Loss_train/delta_cycle": float(delta),
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
                log_dict[f"Loss_train/mmd/{s}"] = float(L_mmd_per[s].detach())
            if delta > 0:
                for (src, tgt), v in L_cyc_per.items():
                    log_dict[f"Loss_train/cycle/{src}_to_{tgt}"] = float(v.detach())
            log_dict.update(_tri_log_metrics("Metric_train", outs, info))

            if batch_idx % 50 == 0:
                print(
                    f"Step [{batch_idx}/{max_len}]\t"
                    f"Loss: {loss.item():.4f} "
                    f"Paired: {L_paired.item():.4f} "
                    f"MMD: {float(L_mmd_total.detach()):.4f} "
                    f"Cyc: {(float(L_cyc_total.detach()) if delta > 0 else 0.0):.4f} "
                    f"γ={gamma:.3f} δ={delta:.3f}"
                )

            wandb.log(log_dict)
            args.global_step += 1
            loss_epoch += loss.item()
            count += 1

        last_targets = targets
        last_outs = outs

    return loss_epoch / max(count, 1), last_targets, last_outs


def train_tri_dynamic_align(paired_train_datasets, dynamic_train_datasets,
                            validation_datasets, dataset_name, dataset_details,
                            model_name, model_details, wandb,
                            run_name='', device='cuda:0', resume_from=None):
    """Joint paired-static + unpaired-dynamic-align training.

    Args:
        paired_train_datasets:  dict {'de': ds, 'ef': ds, 'fd': ds} (static).
        dynamic_train_datasets: dict {'D': ds,  'E': ds,  'F': ds} (unpaired).
        validation_datasets:    dict {split_name: {'de':..., 'ef':..., 'fd':...}}.
        resume_from:            optional path to a checkpoint (e.g. Stage 1
                                dynamic_pretrain) to load weights from.

    Extra hyperparameters (read from `model_details` via getattr, with defaults):
        gamma_mmd_max         (0.5)
        mmd_warmup_epochs     (0)        — linear ramp for γ from 0 → γ_max
        mmd_bandwidths        ((1,2,4,8,16))
        delta_cycle_max       (0.5)
        cycle_warmup_start    (10)       — epoch cycle loss begins
        cycle_warmup_epochs   (10)       — ramp width for δ
        cycle_target          ('mu')     — 'mu' or 'z'
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

    print('=' * 60)
    print('CrossModalVAE3 — paired static + dynamic align (MMD + cycle)')
    print('=' * 60)
    print('Batch:', args.batch_size, 'LR:', args.learning_rate)
    print(f"γ MMD max: {getattr(args, 'gamma_mmd_max', 0.5)} "
          f"(warmup {getattr(args, 'mmd_warmup_epochs', 0)} ep)")
    print(f"δ cycle max: {getattr(args, 'delta_cycle_max', 0.5)} "
          f"(start ep {getattr(args, 'cycle_warmup_start', 10)}, "
          f"warmup {getattr(args, 'cycle_warmup_epochs', 10)} ep)")
    print(f"Cycle target: {getattr(args, 'cycle_target', 'mu')}")

    paired_loaders = {
        k: torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                       shuffle=True, drop_last=True)
        for k, ds in paired_train_datasets.items()
    }
    dynamic_loaders = {
        k: torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                       shuffle=True, drop_last=True)
        for k, ds in dynamic_train_datasets.items()
    }

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
        loss_epoch, last_targets, _ = paired_with_dynamic_align_step(
            args, paired_loaders, dynamic_loaders,
            model, criterion, optimizer, wandb,
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
#       model_details=...,           # add gamma_mmd_max, delta_cycle_max, ...
#       wandb=wandb, run_name=run_name, device=device,
#       resume_from=stage1_ckpt,     # optional dynamic_pretrain weights
#   )
