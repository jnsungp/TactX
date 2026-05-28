import itertools
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
from tqdm import tqdm
from torchvision.utils import make_grid
from joint_embedding_learning.data.datasets import get_all_contrastive_datasets, EpisodeChunkBatchSampler
from joint_embedding_learning.utils.utils import load_optimizer, save_model, get_loss, adjust_learning_rate
from joint_embedding_learning.utils.losses import loss_vector_embedding, cwcl_weight_stats, ContactLabelLookup
from joint_embedding_learning.utils.models_setup import get_model


def _loader_kwargs(args):
    """DataLoader perf kwargs driven by args.num_workers (default 0).

    With num_workers=0 (the default) this returns {} so loaders behave exactly
    as before. With num_workers>0 it parallelizes the per-item torch.load disk
    reads and pins host memory for faster H2D copies. Datasets return CPU
    tensors (the train loop moves batches to device), so worker processes are
    safe — no CUDA tensors cross the fork boundary.
    """
    nw = int(getattr(args, 'num_workers', 0))
    if nw <= 0:
        return {}
    return {
        'num_workers': nw,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': int(getattr(args, 'prefetch_factor', 4)),
    }


def _base_object_names(ds):
    """Peel Subset/random_split wrappers to the base dataset and return its
    .object_names (index->name list used for the object_id in meta)."""
    base = ds
    while hasattr(base, 'dataset'):
        base = base.dataset
    return list(getattr(base, 'object_names', []))



def compute_embedding_metrics(z_i, z_j):
    """Compute embedding-quality metrics on a batch of paired projections.

    Args:
        z_i, z_j: (B, D) projection-head outputs from the two modalities. Diagonal pairs
            (z_i[k], z_j[k]) are positives; off-diagonal cross-modal pairs are negatives.

    Returns:
        dict with pos/neg cosine sim mean+std, gap, and alignment (||z_i - z_j||^2 mean).
    """
    z_i_n = F.normalize(z_i, dim=1)
    z_j_n = F.normalize(z_j, dim=1)
    sim = z_i_n @ z_j_n.t()                                # (B, B)
    B = sim.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=sim.device)
    pos = sim[eye]                                         # (B,)
    neg = sim[~eye]                                        # (B*(B-1),)
    alignment = (z_i_n - z_j_n).pow(2).sum(dim=1).mean()
    pos_std = pos.std(unbiased=False) if B > 1 else torch.zeros((), device=sim.device)
    neg_std = neg.std(unbiased=False) if neg.numel() > 1 else torch.zeros((), device=sim.device)
    return {
        'pos_cos_mean': pos.mean().item(),
        'pos_cos_std' : pos_std.item(),
        'neg_cos_mean': neg.mean().item(),
        'neg_cos_std' : neg_std.item(),
        'gap'         : (pos.mean() - neg.mean()).item(),
        'alignment'   : alignment.item(),
    }

def _use_cwcl(args, model_name):
    return model_name == 'cross_modal' and getattr(args, 'loss', 'infonce') == 'cwcl'


def _use_vae(model_name):
    return model_name == 'cross_modal_vae'


def _use_vae3(model_name):
    return model_name == 'cross_modal_vae3'


def _kl_beta(args):
    """Linear warm-up: 0 -> beta_max over kl_warmup_epochs, then clamped at beta_max."""
    beta_max = float(getattr(args, 'beta_max', 0.0))
    warmup = max(int(getattr(args, 'kl_warmup_epochs', 0)), 1)
    frac = min(1.0, float(args.current_epoch) / warmup)
    return beta_max * frac


def _infonce_alpha(args):
    """Delayed linear warm-up for InfoNCE: 0 until start epoch, then 0 -> 1."""
    start = int(getattr(args, 'infonce_warmup_start', 0))
    warmup = max(int(getattr(args, 'infonce_warmup_epochs', 0)), 1)
    epoch = args.current_epoch
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start) / warmup)


def _dynamic_kl_beta(args):
    """Linear warm-up for Stage 1 KL: 0 -> dynamic_beta_max over dynamic_kl_warmup_epochs."""
    beta_max = float(getattr(args, 'dynamic_beta_max', 0.0))
    warmup = max(int(getattr(args, 'dynamic_kl_warmup_epochs', 0)), 1)
    frac = min(1.0, float(args.current_epoch) / warmup)
    return beta_max * frac


def _compute_loss(args, model_name, criterion, outputs, x_i, x_j, meta):
    """Dispatch loss call.

    outputs is:
      - tuple (h_i, h_j, z_i, z_j) for legacy cross_modal / simclr / barlow_twins
      - dict  from CrossModalVAE for model_name == 'cross_modal_vae'
    Returns (loss_tensor, loss_info_dict).
    """
    if _use_vae(model_name):
        beta = _kl_beta(args)
        losses = criterion(outputs, x_i, x_j, beta=beta)
        return losses['total'], losses
    # legacy path: z_i, z_j tuple
    _, _, z_i, z_j = outputs
    if _use_cwcl(args, model_name):
        return criterion(z_i, z_j, meta), None
    return criterion(z_i, z_j), None


def _unpack_forward(model_name, out):
    """Unify legacy (h_i, h_j, z_i, z_j) tuple and CrossModalVAE dict.

    Returns (h_i, h_j, repr_i, repr_j) where repr_* is used for contrastive
    diagnostics: for VAE we use mu (deterministic, matches InfoNCE target).
    """
    if _use_vae(model_name):
        return out['h_i'], out['h_j'], out['mu_i'], out['mu_j']
    h_i, h_j, z_i, z_j = out
    return h_i, h_j, z_i, z_j


def _prepare_meta(labels, device, model_name):
    """Move cross_modal (tool, episode, frame) labels to device.

    Returns a tuple of three (B,) long tensors, or None for non cross_modal datasets.
    """
    if model_name not in ('cross_modal', 'cross_modal_vae') or labels is None:
        return None
    if not isinstance(labels, (list, tuple)) or len(labels) < 3:
        return None
    tool_ids, episode_ids, frame_ids = labels[0], labels[1], labels[2]
    return (tool_ids.to(device), episode_ids.to(device), frame_ids.to(device))


def train_step(args, train_loaders, model_name, model, criterion, optimizer, scaler, wandb):
    loss_epoch = 0
    loss_vector = 0
    count = 0
    loader_len = len(train_loaders[0]) * len(train_loaders)
    step_lr = args.current_epoch * loader_len
    use_cwcl = _use_cwcl(args, model_name)
    use_vae = _use_vae(model_name)
    for loader in train_loaders:
        for step, ((x_i, x_j), labels) in tqdm(enumerate(loader)):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)
            meta = _prepare_meta(labels, args.device, model_name)

            if args.optimizer == "Adam":
                optimizer.zero_grad()
                out = model(x_i, x_j)
                loss, info = _compute_loss(args, model_name, criterion, out, x_i, x_j, meta)
                loss.backward()
                optimizer.step()

            elif args.optimizer == "LARS":
                adjust_learning_rate(args.epochs, args.batch_size, args.learning_rate_weights, args.learning_rate_biases, optimizer, loader_len, step_lr)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    out = model(x_i, x_j)
                    loss, info = _compute_loss(args, model_name, criterion, out, x_i, x_j, meta)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                raise NotImplementedError

            with torch.no_grad():
                h_i, h_j, repr_i, repr_j = _unpack_forward(model_name, out)
                loss_vector_0 = loss_vector_embedding(h_i, h_j)
                m = compute_embedding_metrics(repr_i, repr_j)

                if step % 50 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")
                    print(f"Step [{step}/{len(loader)}]\t Loss Vector: {loss_vector_0.item()}")

                log_dict = {
                    "Loss_train/Model_Loss_batch": loss.item(),
                    "Loss_train/embedding_diff_batch": loss_vector_0.item(),
                    "Metric_train/pos_cos_mean": m['pos_cos_mean'],
                    "Metric_train/pos_cos_std" : m['pos_cos_std'],
                    "Metric_train/neg_cos_mean": m['neg_cos_mean'],
                    "Metric_train/neg_cos_std" : m['neg_cos_std'],
                    "Metric_train/gap"         : m['gap'],
                    "Metric_train/alignment"   : m['alignment'],
                    "Epoch_batch": args.current_epoch,
                }
                if use_cwcl and meta is not None:
                    stats = cwcl_weight_stats(
                        meta[0], meta[1], meta[2],
                        sigma=getattr(args, 'cwcl_sigma', 5.0),
                        same_episode_only=getattr(args, 'cwcl_same_episode_only', True),
                    )
                    for k, v in stats.items():
                        log_dict[f"Metric_train/{k}"] = v
                if use_vae and info is not None:
                    for k in ('infonce', 'recon', 'kl',
                              'recon_ii', 'recon_ji', 'recon_jj', 'recon_ij',
                              'kl_i', 'kl_j', 'beta'):
                        v = info.get(k)
                        if v is None:
                            continue
                        log_dict[f"Loss_train/{k}"] = v.item() if torch.is_tensor(v) else float(v)
                wandb.log(log_dict)
                args.global_step += 1

                loss_epoch += loss.item()
                loss_vector += loss_vector_0.item()
                count += 1
                step_lr += 1

    loss_epoch /= count
    loss_vector /= count
    return loss_epoch, loss_vector, x_i, x_j

def validation_step(args, val_loaders, model, criterion, wandb, val_name = "Val", model_name='cross_modal'):
    loss_epoch = 0
    loss_vector = 0
    count = 0
    use_vae = _use_vae(model_name)

    model.eval()
    with torch.no_grad():
        for val_loader in val_loaders:
            for step, ((x_i, x_j), labels) in tqdm(enumerate(val_loader)):
                x_i = x_i.to(args.device)
                x_j = x_j.to(args.device)
                meta = _prepare_meta(labels, args.device, model_name)

                out = model(x_i, x_j)
                loss, info = _compute_loss(args, model_name, criterion, out, x_i, x_j, meta)
                h_i, h_j, repr_i, repr_j = _unpack_forward(model_name, out)
                loss_vector_0 = loss_vector_embedding(h_i, h_j)
                m = compute_embedding_metrics(repr_i, repr_j)

                if step % 50 == 0:
                    print(f"Step [{step}/{len(val_loader)}]\t" + val_name + "Loss: {loss.item()}")
                    print(f"Step [{step}/{len(val_loader)}]\t" + val_name + "Loss Vector: {loss_vector_0.item()}")

                log_dict = {
                    "Loss_" + val_name + "/Model_Loss_batch": loss.item(),
                    "Loss_" + val_name + "/embedding_diff_batch": loss_vector_0.item(),
                    f"Metric_{val_name}/pos_cos_mean": m['pos_cos_mean'],
                    f"Metric_{val_name}/pos_cos_std" : m['pos_cos_std'],
                    f"Metric_{val_name}/neg_cos_mean": m['neg_cos_mean'],
                    f"Metric_{val_name}/neg_cos_std" : m['neg_cos_std'],
                    f"Metric_{val_name}/gap"         : m['gap'],
                    f"Metric_{val_name}/alignment"   : m['alignment'],
                    "Epoch": args.current_epoch,
                }
                if use_vae and info is not None:
                    for k in ('infonce', 'recon', 'kl',
                              'recon_ii', 'recon_ji', 'recon_jj', 'recon_ij',
                              'kl_i', 'kl_j', 'beta'):
                        v = info.get(k)
                        if v is None:
                            continue
                        log_dict[f"Loss_{val_name}/{k}"] = v.item() if torch.is_tensor(v) else float(v)
                wandb.log(log_dict)

                loss_epoch += loss.item()
                loss_vector += loss_vector_0.item()
                count += 1

    model.train()
    loss_epoch /= count
    loss_vector /= count
    return loss_epoch, loss_vector, x_i, x_j


def _tri_pair_metrics(out, prefix):
    """Compute pos/neg cosine and alignment metrics for one tri-pair forward."""
    return compute_embedding_metrics(out["mu_a"], out["mu_b"])


@torch.no_grad()
def _transitive_df_metric(out_de, out_ef):
    """Indirect D-F alignment proxy.

    For each ef_pair sample, find the nearest de_pair sample by cosine of mu_E
    (the bridge sensor present in both pairs), then align the corresponding
    mu_F (from ef_pair) with mu_D (from de_pair).

    Returns the diagonal-vs-offdiagonal cosine gap on the implied (mu_F, mu_D)
    correspondence; > 0 means transitive D-F alignment is emerging.
    """
    mu_E_de = out_de["mu_b"]   # de pair: a=D, b=E
    mu_D_de = out_de["mu_a"]
    mu_E_ef = out_ef["mu_a"]   # ef pair: a=E, b=F
    mu_F_ef = out_ef["mu_b"]

    a = F.normalize(mu_E_ef, dim=1)
    b = F.normalize(mu_E_de, dim=1)
    sim_E = a @ b.t()  # (B, B)
    nn_idx = sim_E.argmax(dim=1)  # for each ef sample, idx of most similar de sample
    mu_D_match = mu_D_de[nn_idx]

    z_F = F.normalize(mu_F_ef, dim=1)
    z_D = F.normalize(mu_D_match, dim=1)
    sim = z_F @ z_D.t()
    B = sim.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=sim.device)
    pos = sim[eye]
    neg = sim[~eye]
    return {
        'df_pos_cos_mean': pos.mean().item(),
        'df_neg_cos_mean': neg.mean().item(),
        'df_gap'         : (pos.mean() - neg.mean()).item(),
    }


def _tri_zip_loaders(train_loaders):
    """Yield aligned batches from {de, ef, fd} loaders.

    Each epoch length = max(len(loader_i)). Shorter loaders are cycled with
    itertools.cycle so all data from the longest loader is seen each epoch.
    """
    max_len = max(len(l) for l in train_loaders.values())
    iters = {k: iter(itertools.cycle(l)) if len(l) < max_len else iter(l)
             for k, l in train_loaders.items()}
    for _ in range(max_len):
        yield {k: next(it) for k, it in iters.items()}


def _tri_log_metrics(prefix, outs, info):
    """Build wandb log dict with cosine metrics per pair grouped under shared prefix.

    For each metric kind in {cos_pos, cos_neg, cos_gap, alignment}, all four pair
    series ('de', 'ef', 'fd', 'df_transitive') are emitted under the same
    sub-prefix so wandb plots them on one chart automatically:

        {prefix}/cos_pos/de, {prefix}/cos_pos/ef, {prefix}/cos_pos/fd,
        {prefix}/cos_pos/df_transitive
        {prefix}/cos_neg/...
        {prefix}/cos_gap/...
        {prefix}/alignment/...

    Per-pair losses live under "Loss_..." groups (caller builds them).
    """
    log = {}
    for p in ('de', 'ef', 'fd'):
        m = compute_embedding_metrics(outs[p]["mu_a"], outs[p]["mu_b"])
        log[f"{prefix}/cos_pos/{p}"] = m['pos_cos_mean']
        log[f"{prefix}/cos_neg/{p}"] = m['neg_cos_mean']
        log[f"{prefix}/cos_gap/{p}"] = m['gap']
        log[f"{prefix}/alignment/{p}"] = m['alignment']

    tm = _transitive_df_metric(outs['de'], outs['ef'])
    log[f"{prefix}/cos_pos/df_transitive"] = tm['df_pos_cos_mean']
    log[f"{prefix}/cos_neg/df_transitive"] = tm['df_neg_cos_mean']
    log[f"{prefix}/cos_gap/df_transitive"] = tm['df_gap']

    log[f"{prefix}/cos_gap_avg_direct"] = (
        log[f"{prefix}/cos_gap/de"] + log[f"{prefix}/cos_gap/ef"]
        + log[f"{prefix}/cos_gap/fd"]
    ) / 3.0
    return log


def tri_train_step(args, train_loaders, model, criterion, optimizer, wandb,
                   scaler=None):
    loss_epoch = 0
    count = 0
    max_len = max(len(l) for l in train_loaders.values())
    last_targets = None
    last_outs = None
    use_amp = scaler is not None

    for batch_idx, batches in enumerate(tqdm(_tri_zip_loaders(train_loaders), total=max_len)):
        targets = {}
        meta = {}
        for pair_key, ((x_a, x_b), labels) in batches.items():
            x_a = x_a.to(args.device)
            x_b = x_b.to(args.device)
            targets[pair_key] = (x_a, x_b)
            meta[pair_key] = tuple(t.to(args.device) for t in labels)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            outs = {p: model.forward_pair(targets[p][0], targets[p][1], p)
                    for p in ('de', 'ef', 'fd')}
            beta = _kl_beta(args)
            alpha = _infonce_alpha(args)
            info = criterion(outs, targets, beta=beta, alpha=alpha, meta=meta)
            loss = info['total']
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            log_dict = {
                "Loss_train/Model_Loss_batch": loss.item(),
                "Loss_train/infonce_total": info['infonce'].item(),
                "Loss_train/recon_total": info['recon'].item(),
                "Loss_train/kl_total": info['kl'].item() if torch.is_tensor(info['kl']) else float(info['kl']),
                "Loss_train/beta": float(info['beta']),
                "Loss_train/alpha_infonce": float(info['alpha_infonce']),
                "Epoch_batch": args.current_epoch,
            }
            for p in ('de', 'ef', 'fd'):
                log_dict[f"Loss_train/infonce/{p}"] = info[f"infonce_{p}"].item()
                log_dict[f"Loss_train/recon/{p}"] = info[f"recon_{p}"].item()
            for k in info:
                if (k.startswith("kl_") or k.startswith("cwcl_")
                        or k.startswith("debiased_") or k.startswith("neighbor_")
                        or k.startswith("zero_supcon_")):
                    log_dict[f"Loss_train/{k}"] = info[k].item() if torch.is_tensor(info[k]) else float(info[k])

            log_dict.update(_tri_log_metrics("Metric_train", outs, info))

            if batch_idx % 50 == 0:
                print(f"Step [{batch_idx}/{max_len}]\t Loss: {loss.item():.4f} "
                      f"InfoNCE: {info['infonce'].item():.4f} "
                      f"Recon: {info['recon'].item():.4f} "
                      f"KL: {info['kl'].item():.4f}")

            wandb.log(log_dict)
            args.global_step += 1
            loss_epoch += loss.item()
            count += 1

        last_targets = targets
        last_outs = outs

    loss_epoch /= max(count, 1)
    return loss_epoch, last_targets, last_outs


def tri_validation_step(args, val_loaders, model, criterion, wandb,
                        val_name='Val', use_amp=False):
    if not val_loaders:
        return None, None, None

    loss_epoch = 0
    count = 0
    model.eval()
    last_targets = None
    last_outs = None

    with torch.no_grad():
        max_len = max(len(l) for l in val_loaders.values())
        for batches in tqdm(_tri_zip_loaders(val_loaders), total=max_len):
            targets = {}
            meta = {}
            for pair_key, ((x_a, x_b), labels) in batches.items():
                x_a = x_a.to(args.device)
                x_b = x_b.to(args.device)
                targets[pair_key] = (x_a, x_b)
                meta[pair_key] = tuple(t.to(args.device) for t in labels)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outs = {p: model.forward_pair(targets[p][0], targets[p][1], p)
                        for p in ('de', 'ef', 'fd')}
                beta = _kl_beta(args)
                alpha = _infonce_alpha(args)
                info = criterion(outs, targets, beta=beta, alpha=alpha, meta=meta)
                loss = info['total']

            log_dict = {
                f"Loss_{val_name}/Model_Loss_batch": loss.item(),
                f"Loss_{val_name}/infonce_total": info['infonce'].item(),
                f"Loss_{val_name}/recon_total": info['recon'].item(),
                f"Loss_{val_name}/kl_total": info['kl'].item() if torch.is_tensor(info['kl']) else float(info['kl']),
                f"Loss_{val_name}/alpha_infonce": float(info['alpha_infonce']),
                "Epoch": args.current_epoch,
            }
            for p in ('de', 'ef', 'fd'):
                log_dict[f"Loss_{val_name}/infonce/{p}"] = info[f"infonce_{p}"].item()
                log_dict[f"Loss_{val_name}/recon/{p}"] = info[f"recon_{p}"].item()

            log_dict.update(_tri_log_metrics(f"Metric_{val_name}", outs, info))

            wandb.log(log_dict)
            loss_epoch += loss.item()
            count += 1
            last_targets = targets
            last_outs = outs

    model.train()
    loss_epoch /= max(count, 1)
    return loss_epoch, last_targets, last_outs


def train_tri(train_datasets, validation_datasets, dataset_name, dataset_details,
              model_name, model_details, wandb, run_name='', device='cuda:0',
              resume_from=None):
    """Joint pair-only training loop for cross_modal_vae3.

    train_datasets / validation_datasets: dict {'de': ..., 'ef': ..., 'fd': ...}
    resume_from: optional path to a Stage 1 checkpoint to load weights from.
    """
    model_parser = argparse.ArgumentParser(description="Model-Configuration")
    model_parser.add_argument('--dataset_name', type=str, default='dataset_2')
    model_parser.add_argument('--model_name', type=str, default='cross_modal_vae3')
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
    print('CrossModalVAE3 (tri-modal pair-only joint training)')
    print('=' * 60)
    print('Batch Size:', args.batch_size)
    print('Optimizer:', args.optimizer, 'lr=', args.learning_rate)
    print('Temperature:', args.temperature)
    print('Magnetic Input Dim:', args.magnetic_input_dim)
    print('Magnetic Hidden Dims:', args.magnetic_hidden_dims)
    print('FlexiTac in (h, w):', args.flexitac_in_h, args.flexitac_in_w)
    print('FlexiTac encoder ch:', args.flexitac_encoder_channels)
    print('FlexiTac decoder ch:', args.flexitac_decoder_channels)
    print('Lambda Recon:', getattr(args, 'lambda_recon', 1.0))
    print('Beta Max:', getattr(args, 'beta_max', 0.0))
    print('KL Warmup Epochs:', getattr(args, 'kl_warmup_epochs', 0))
    print('InfoNCE Warmup Start:', getattr(args, 'infonce_warmup_start', 0))
    print('InfoNCE Warmup Epochs:', getattr(args, 'infonce_warmup_epochs', 0))

    # Episode-aware batching: required by CWCL / neighbor so same-episode
    # frames co-occur in a batch (auto-on for loss in {'cwcl', 'neighbor'};
    # opt-in elsewhere via use_episode_sampler). Otherwise standard shuffled
    # DataLoaders.
    _use_ep_sampler = (getattr(args, 'use_episode_sampler', False)
                       or getattr(args, 'loss', 'infonce') in ('cwcl', 'neighbor'))
    _chunk = int(getattr(args, 'cwcl_chunk_size', 8))

    def _make_loader(ds, seed):
        if _use_ep_sampler:
            sampler = EpisodeChunkBatchSampler(
                ds, batch_size=args.batch_size, chunk_size=_chunk,
                drop_last=True, seed=seed,
            )
            return torch.utils.data.DataLoader(
                ds, batch_sampler=sampler, **_loader_kwargs(args)
            )
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
            **_loader_kwargs(args)
        )

    if _use_ep_sampler:
        print(f"Episode-aware batch sampler ON (chunk_size={_chunk})")

    train_loaders = {
        k: _make_loader(ds, seed=i)
        for i, (k, ds) in enumerate(train_datasets.items())
    }
    val_loaders_array = {}
    for split_key, ds_dict in validation_datasets.items():
        if not ds_dict:
            continue
        loaders = {}
        for pi, (pair_key, ds) in enumerate(ds_dict.items()):
            if len(ds) == 0:
                continue
            loaders[pair_key] = _make_loader(ds, seed=100 + pi)
        # Need all 3 pairs present (de/ef/fd) for tri-modal joint validation.
        if len(loaders) == len(ds_dict) and len(loaders) > 0:
            val_loaders_array[split_key] = loaders

    model = get_model(model_name, model_details)
    model = model.to(args.device)

    resume_from = getattr(args, 'resume_from', None)
    if resume_from:
        state = torch.load(resume_from, map_location=args.device, weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Stage 2] Loaded weights from: {resume_from}")
        if missing:
            print(f"  missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    optimizer, scheduler, _ = load_optimizer(args, model)
    criterion = get_loss(args, model_name)

    # Debiased / zero_supcon InfoNCE need per-frame contact labels. Build a
    # (object_id, episode, frame) -> no-contact lookup once, keyed with each
    # pair's base-dataset object_names so the keys line up with `meta`.
    _loss_type = getattr(args, 'loss', 'infonce')
    if _loss_type in ('debiased', 'zero_supcon'):
        label_root = getattr(args, 'contact_label_root', None)
        if not label_root:
            raise ValueError(
                f"loss='{_loss_type}' requires `contact_label_root` in "
                f"model_details (set it via --contact_label_root)."
            )
        object_names_per_pair = {
            p: _base_object_names(train_datasets[p]) for p in ('de', 'ef', 'fd')
        }
        contact_lookup = ContactLabelLookup(label_root, object_names_per_pair)
        criterion.set_contact_lookup(contact_lookup)
        print(f"[{_loss_type}] contact labels from: {label_root}")
        print(f"[{_loss_type}] labels loaded per pair: "
              f"{ {p: contact_lookup.n_labels[p] for p in ('de','ef','fd')} }")
        if _loss_type == 'debiased':
            print(f"[debiased] masks -> "
                  f"same_episode={criterion.mask_same_episode}, "
                  f"zero_zero={criterion.mask_zero_zero}")
        for p in ('de', 'ef', 'fd'):
            if contact_lookup.n_labels[p] == 0:
                print(f"[{_loss_type}] WARNING: 0 labels found for pair '{p}' "
                      f"under {label_root}/{p}_pair/ — check "
                      f"--contact_label_root and that --objects match the "
                      f"label folders.")

    use_amp = bool(getattr(args, 'amp', False)) and args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed-precision training (AMP) enabled")

    args.global_step = 0
    args.current_epoch = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epochs"):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch, last_targets, _last_outs = tri_train_step(
            args, train_loaders, model, criterion, optimizer, wandb,
            scaler=scaler,
        )
        wandb.log({"Loss_train/Model_Loss": loss_epoch,
                   "Misc/learning_rate": lr,
                   "Epoch": epoch})

        if scheduler:
            scheduler.step()

        with torch.no_grad():
            if epoch % 10 == 0:
                # Periodic kept snapshot: every 50 epochs once past epoch 100,
                # so long runs are recoverable (model.pt itself is overwritten).
                snap = f"ep{epoch}" if (epoch >= 100 and epoch % 50 == 0) else None
                save_model(model, dataset_name, dataset_type, model_type,
                           dataset_details, model_details, run_name=run_name,
                           tag=snap)
                if last_targets is not None:
                    x_D, _ = last_targets['de']
                    grid = make_grid(x_D[:36], nrow=6, normalize=True)
                    wandb.log({"Image_Daimon": [wandb.Image(grid)], "Epoch": epoch})

            for key, val_loaders in val_loaders_array.items():
                val_loss_epoch, _, _ = tri_validation_step(
                    args, val_loaders, model, criterion, wandb, val_name=key,
                    use_amp=use_amp,
                )
                if val_loss_epoch is not None:
                    wandb.log({f"Loss_{key}/Model_Loss": val_loss_epoch,
                               "Epoch": epoch})

            args.current_epoch += 1

    save_model(model, dataset_name, dataset_type, model_type,
               dataset_details, model_details, run_name=run_name)


# ---------------------------------------------------------------------------
# Stage 1: Dynamic self-recon pretrain (unpaired, per-modality)
# ---------------------------------------------------------------------------

from joint_embedding_learning.utils.losses import kl_divergence_standard

_SENSOR_KEYS = ('D', 'E', 'F')


def _shape_match_recon(recon, target):
    """Reconcile potential singleton channel mismatch (mirrors loss util)."""
    if recon.shape == target.shape:
        return recon
    if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
        return recon.squeeze(1)
    if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
        return recon.unsqueeze(1)
    return recon


def dynamic_pretrain_step(args, train_loaders, model, optimizer, wandb):
    """One epoch of Stage 1 dynamic self-recon pretrain.

    train_loaders: dict {'D': loader_D, 'E': loader_E, 'F': loader_F}
    Each loader yields (x, [episode_id, frame_id]).
    """
    loss_epoch = 0.0
    count = 0
    max_len = max(len(l) for l in train_loaders.values())
    last_inputs = None

    for batch_idx, batches in enumerate(tqdm(_tri_zip_loaders(train_loaders), total=max_len)):
        inputs = {}
        for sensor_key, (x, _labels) in batches.items():
            inputs[sensor_key] = x.to(args.device)

        optimizer.zero_grad()
        outs = {s: model.forward_self(inputs[s], s) for s in _SENSOR_KEYS}

        beta = _dynamic_kl_beta(args)
        lambda_recon = float(getattr(args, 'lambda_recon', 1.0))

        recon_per_sensor = {}
        kl_per_sensor = {}
        L_recon_total = 0.0
        L_kl_total = 0.0
        for s in _SENSOR_KEYS:
            x = inputs[s]
            recon = _shape_match_recon(outs[s]['recon'], x)
            r = F.l1_loss(recon, x)
            kl = kl_divergence_standard(outs[s]['mu'], outs[s]['logvar'])
            L_recon_total = L_recon_total + r
            L_kl_total = L_kl_total + kl
            recon_per_sensor[s] = r.detach()
            kl_per_sensor[s] = kl.detach()

        loss = lambda_recon * L_recon_total + beta * L_kl_total
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            log_dict = {
                "Loss_dynpre/Model_Loss_batch": loss.item(),
                "Loss_dynpre/recon_total": float(L_recon_total.detach()),
                "Loss_dynpre/kl_total": float(L_kl_total.detach()),
                "Loss_dynpre/beta": float(beta),
                "Epoch_batch": args.current_epoch,
            }
            for s in _SENSOR_KEYS:
                log_dict[f"Loss_dynpre/recon/{s}"] = float(recon_per_sensor[s])
                log_dict[f"Loss_dynpre/kl/{s}"] = float(kl_per_sensor[s])

            if batch_idx % 50 == 0:
                print(f"Step [{batch_idx}/{max_len}]\t Loss: {loss.item():.4f} "
                      f"Recon: {float(L_recon_total.detach()):.4f} "
                      f"KL: {float(L_kl_total.detach()):.4f} beta={beta:.4f}")

            wandb.log(log_dict)
            args.global_step += 1
            loss_epoch += loss.item()
            count += 1

        last_inputs = inputs

    loss_epoch /= max(count, 1)
    return loss_epoch, last_inputs


def dynamic_pretrain_val_step(args, val_loaders, model, wandb, val_name='Val'):
    """Validation pass with the same self-recon loss (no gradient)."""
    if not val_loaders:
        return None

    loss_epoch = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        max_len = max(len(l) for l in val_loaders.values())
        for batches in tqdm(_tri_zip_loaders(val_loaders), total=max_len):
            inputs = {}
            for sensor_key, (x, _labels) in batches.items():
                inputs[sensor_key] = x.to(args.device)
            outs = {s: model.forward_self(inputs[s], s) for s in _SENSOR_KEYS}

            beta = _dynamic_kl_beta(args)
            lambda_recon = float(getattr(args, 'lambda_recon', 1.0))
            L_recon_total = 0.0
            L_kl_total = 0.0
            recon_per_sensor = {}
            for s in _SENSOR_KEYS:
                x = inputs[s]
                recon = _shape_match_recon(outs[s]['recon'], x)
                r = F.l1_loss(recon, x)
                kl = kl_divergence_standard(outs[s]['mu'], outs[s]['logvar'])
                L_recon_total = L_recon_total + r
                L_kl_total = L_kl_total + kl
                recon_per_sensor[s] = float(r.detach())

            loss = lambda_recon * L_recon_total + beta * L_kl_total
            log_dict = {
                f"Loss_{val_name}/Model_Loss_batch": loss.item(),
                f"Loss_{val_name}/recon_total": float(L_recon_total.detach()),
                f"Loss_{val_name}/kl_total": float(L_kl_total.detach()),
                "Epoch": args.current_epoch,
            }
            for s in _SENSOR_KEYS:
                log_dict[f"Loss_{val_name}/recon/{s}"] = recon_per_sensor[s]
            wandb.log(log_dict)
            loss_epoch += loss.item()
            count += 1
    model.train()
    return loss_epoch / max(count, 1)


def train_tri_dynamic_pretrain(train_datasets, validation_datasets, dataset_name,
                               dataset_details, model_name, model_details, wandb,
                               run_name='', device='cuda:0'):
    """Stage 1 training loop: dynamic self-recon only (unpaired).

    train_datasets / validation_datasets: dict {'D': ds, 'E': ds, 'F': ds}.
    Epochs / KL warmup follow dynamic_epochs / dynamic_kl_warmup_epochs.
    Saves checkpoint under dataset_type='cross_modal_tri_dynamic'.
    """
    model_parser = argparse.ArgumentParser(description="Model-Configuration")
    model_parser.add_argument('--dataset_name', type=str, default='dataset_cross_modal_tri_dynamic')
    model_parser.add_argument('--model_name', type=str, default='cross_modal_vae3')
    model_parser.add_argument("--device", default=device, type=str)
    model_parser.add_argument("--run_name", type=str, default=run_name)
    model_parser.add_argument("--debug", action='store_true')
    args = model_parser.parse_args([])
    for key, value in model_details.items():
        setattr(args, key, value)

    # Override epoch count with dynamic-specific value if provided.
    dynamic_epochs = int(getattr(args, 'dynamic_epochs', args.epochs))
    args.epochs = dynamic_epochs

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_type = dataset_details['dataset_type']
    model_type = model_name

    print('=' * 60)
    print('CrossModalVAE3 Stage 1: Dynamic Self-Recon Pretrain')
    print('=' * 60)
    print('Batch Size:', args.batch_size)
    print('Optimizer:', args.optimizer, 'lr=', args.learning_rate)
    print('Lambda Recon:', getattr(args, 'lambda_recon', 1.0))
    print('Dynamic Beta Max:', getattr(args, 'dynamic_beta_max', 0.0))
    print('Dynamic KL Warmup Epochs:', getattr(args, 'dynamic_kl_warmup_epochs', 0))
    print('Dynamic Epochs:', dynamic_epochs)

    train_loaders = {
        k: torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                       shuffle=True, drop_last=True,
                                       **_loader_kwargs(args))
        for k, ds in train_datasets.items()
    }
    val_loaders = None
    if validation_datasets:
        # validation_datasets here is a single dict {'D': ds, 'E': ds, 'F': ds}.
        loaders = {}
        for sensor_key, ds in validation_datasets.items():
            if len(ds) == 0:
                continue
            loaders[sensor_key] = torch.utils.data.DataLoader(
                ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                **_loader_kwargs(args)
            )
        if len(loaders) == len(_SENSOR_KEYS):
            val_loaders = loaders

    model = get_model(model_name, model_details)
    model = model.to(args.device)

    optimizer, scheduler, _ = load_optimizer(args, model)

    args.global_step = 0
    args.current_epoch = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epochs"):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch, last_inputs = dynamic_pretrain_step(
            args, train_loaders, model, optimizer, wandb
        )
        wandb.log({"Loss_dynpre/Model_Loss": loss_epoch,
                   "Misc/learning_rate": lr,
                   "Epoch": epoch})

        if scheduler:
            scheduler.step()

        with torch.no_grad():
            if epoch % 10 == 0:
                save_model(model, dataset_name, dataset_type, model_type,
                           dataset_details, model_details, run_name=run_name)
                if last_inputs is not None and 'D' in last_inputs:
                    x_D = last_inputs['D']
                    grid = make_grid(x_D[:36], nrow=6, normalize=True)
                    wandb.log({"Image_Daimon_dynpre": [wandb.Image(grid)], "Epoch": epoch})

            if val_loaders is not None:
                val_loss = dynamic_pretrain_val_step(
                    args, val_loaders, model, wandb, val_name='Val_dynpre'
                )
                if val_loss is not None:
                    wandb.log({"Loss_Val_dynpre/Model_Loss": val_loss, "Epoch": epoch})

            args.current_epoch += 1

    save_model(model, dataset_name, dataset_type, model_type,
               dataset_details, model_details, run_name=run_name)


def train(train_datasets, validation_datasets, dataset_name, dataset_details, model_name, model_details, wandb, run_name = '', device='cuda:0'):
    model_parser = argparse.ArgumentParser(description="Model-Configuration")
    model_parser.add_argument('--dataset_name', type=str, default='dataset_2', help='Name of the dataset')
    model_parser.add_argument('--model_name', type=str, default='simclr', help='Name of the model')
    model_parser.add_argument("--device", default=device, type=str, help="Device to run the model on")
    model_parser.add_argument("--run_name", type=str, default=run_name, help='Name of the run')
    model_parser.add_argument("--debug", action='store_true', help='Debug mode')
    args = model_parser.parse_args([])

    for key, value in model_details.items():
        setattr(args, key, value)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_type = dataset_details['dataset_type']
    model_type = model_name

    print('Batch Size:', args.batch_size)
    print('Optimizer:', args.optimizer)
    if args.optimizer == "Adam":
        print('Learning Rate:', args.learning_rate)
    

    if args.model_name == 'simclr':
        print('Temperature:', args.temperature)
    elif args.model_name == 'barlow_twins':
        print('Lambda:', args.lambd)
    elif args.model_name == 'cross_modal':
        print('Temperature:', args.temperature)
        print('Magnetic Input Dim:', args.magnetic_input_dim)
        print('Magnetic Hidden Dims:', args.magnetic_hidden_dims)
        loss_type = getattr(args, 'loss', 'infonce')
        print('Loss:', loss_type)
        if loss_type == 'cwcl':
            print('CWCL sigma:', getattr(args, 'cwcl_sigma', 5.0))
            print('CWCL same_episode_only:', getattr(args, 'cwcl_same_episode_only', True))
    elif args.model_name == 'cross_modal_vae':
        print('Temperature:', args.temperature)
        print('Magnetic Input Dim:', args.magnetic_input_dim)
        print('Magnetic Hidden Dims:', args.magnetic_hidden_dims)
        print('Lambda Recon:', getattr(args, 'lambda_recon', 1.0))
        print('Beta Max:', getattr(args, 'beta_max', 0.0))
        print('KL Warmup Epochs:', getattr(args, 'kl_warmup_epochs', 0))
        print('Image Decoder Channels:', getattr(args, 'image_decoder_channels', [256, 128, 64, 32]))
        print('Magnetic Decoder Hidden:', getattr(args, 'magnetic_decoder_hidden', [256, 128, 64]))
    
    train_loaders = []
    for dataset in train_datasets:
        train_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **_loader_kwargs(args)))

    val_loaders_array = {}
    for key, value in validation_datasets.items():
        val_loaders = []
        for dataset in value:
            if len(dataset) == 0:
                continue
            val_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **_loader_kwargs(args)))
        if val_loaders:
            val_loaders_array[key] = val_loaders

    # initialize ResNet
    model = get_model(model_name, model_details)
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler, scaler = load_optimizer(args, model)
    criterion = get_loss(args, model_name)

    args.global_step = 0
    args.current_epoch = 0
    
    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epochs"):
        lr = optimizer.param_groups[0]["lr"]

        loss_epoch, loss_vector, x_i, x_j = train_step(args, train_loaders, model_name, model, criterion, optimizer, scaler, wandb)
        wandb.log({"Loss_train/Model_Loss": loss_epoch, "Loss_train/embedding_diff":loss_vector, "Misc/learning_rate": lr, "Epoch": epoch})

        if scheduler:
            scheduler.step()

        with torch.no_grad():

            if epoch % 10 == 0:
                save_model(model, dataset_name, dataset_type, model_type, dataset_details, model_details, run_name=run_name)
                if model_name in ('cross_modal', 'cross_modal_vae'):
                    grid = make_grid(x_i[:36], nrow=6, normalize=True)
                    wandb.log({"Image_Daimon": [wandb.Image(grid)], "Epoch": epoch})
                else:
                    grid_i = make_grid(x_i[:36], nrow=6, normalize=True)
                    grid_j = make_grid(x_j[:36], nrow=6, normalize=True)
                    wandb.log({"Image": [wandb.Image(grid_i), wandb.Image(grid_j)], "Epoch": epoch})

            for key, val_loader in val_loaders_array.items():
                val_loss_epoch, val_loss_vector, x_i, x_j = validation_step(args, val_loader, model, criterion, wandb, key, model_name=model_name)
                wandb.log({"Loss_" + key + "/Model_Loss": val_loss_epoch, "Loss_" + key + "/embedding_diff": val_loss_vector, "Epoch": epoch})

            args.current_epoch += 1

    save_model(model, dataset_name, dataset_type, model_type, dataset_details, model_details, run_name=run_name)