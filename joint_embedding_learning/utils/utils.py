import os
import math
import torch
import yaml
from simclr.modules import NT_Xent
from joint_embedding_learning.utils.losses import CWCL, CrossModalVAELoss, CrossModalVAE3Loss, CrossModalVAE3UniforceLoss, CrossModalVAE3CWCLLoss, CrossModalVAE3NeighborLoss, CrossModalVAE3DebiasedLoss, CrossModalVAE3ZeroSupConLoss, CrossModalCLR3Loss


# ---------------------------------------------------------------------------
# LARS optimizer + cosine LR schedule.
# Vendored verbatim from the (removed) barlow_twins package so the legacy LARS
# training path keeps working without that dependency. Only reached when a
# model config sets optimizer: "LARS"; the cross_modal_vae3 / cross_modal_clr3
# configs use Adam, so these are dormant for the documented training commands.
# ---------------------------------------------------------------------------
def adjust_learning_rate(epochs, batch_size, learning_rate_weights,
                         learning_rate_biases, optimizer, loader_len, step):
    max_steps = epochs * loader_len
    warmup_steps = 10 * loader_len
    base_lr = batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * learning_rate_biases


class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def load_optimizer(args, model):
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # TODO: LARS
        scheduler = None
        scaler = None
    elif args.optimizer == "LARS":
        scheduler = None
        param_weights = []
        param_biases = []
        for param in model.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=True,
                        lars_adaptation_filter=True)
        
        scaler = torch.cuda.amp.GradScaler()
    else:
        raise NotImplementedError
    
    return optimizer, scheduler, scaler

def get_loss(args, model_name):
    if model_name == 'simclr':
        criterion = NT_Xent(args.batch_size, args.temperature, 1)
    elif model_name == 'cross_modal':
        loss_type = getattr(args, 'loss', 'infonce')
        if loss_type == 'cwcl':
            criterion = CWCL(
                temperature=args.temperature,
                sigma=getattr(args, 'cwcl_sigma', 5.0),
                same_episode_only=getattr(args, 'cwcl_same_episode_only', True),
            )
        elif loss_type == 'infonce':
            criterion = NT_Xent(args.batch_size, args.temperature, 1)
        else:
            raise ValueError(f"Unknown loss '{loss_type}' for cross_modal (expected 'infonce' or 'cwcl')")
    elif model_name == 'cross_modal_vae':
        criterion = CrossModalVAELoss(
            batch_size=args.batch_size,
            temperature=args.temperature,
            lambda_recon=getattr(args, 'lambda_recon', 1.0),
        )
    elif model_name == 'cross_modal_vae3':
        # Alignment objective: 'infonce' (default, contrastive),
        # 'uniforce'/'l2' (plain L2 pull), or 'cwcl' (soft-weighted
        # contrastive — needs an episode-aware batch sampler).
        loss_type = getattr(args, 'loss', 'infonce')
        if loss_type in ('uniforce', 'l2'):
            criterion = CrossModalVAE3UniforceLoss(
                batch_size=args.batch_size,
                temperature=args.temperature,
                lambda_recon=getattr(args, 'lambda_recon', 1.0),
            )
        elif loss_type == 'cwcl':
            criterion = CrossModalVAE3CWCLLoss(
                batch_size=args.batch_size,
                temperature=args.temperature,
                lambda_recon=getattr(args, 'lambda_recon', 1.0),
                cwcl_sigma=getattr(args, 'cwcl_sigma', 5.0),
                cwcl_same_episode_only=getattr(args, 'cwcl_same_episode_only', True),
            )
        elif loss_type == 'neighbor':
            # NT-Xent (matched cross-sensor) + auxiliary intra-modal cosine
            # smoothness on ±1 same-(object, episode) frames. Default weight
            # 0.05 — high enough to smooth, low enough to keep retrieval.
            # Needs an episode-aware batch sampler — train_tri auto-on.
            criterion = CrossModalVAE3NeighborLoss(
                batch_size=args.batch_size,
                temperature=args.temperature,
                lambda_recon=getattr(args, 'lambda_recon', 1.0),
                neighbor_weight=getattr(args, 'neighbor_weight', 0.05),
                same_episode_only=getattr(args, 'neighbor_same_episode_only', True),
            )
        elif loss_type == 'debiased':
            # False-negative-masked InfoNCE: different-episode-only negatives
            # + per-sensor zero-zero mask. Needs per-frame contact labels —
            # train_tri attaches a ContactLabelLookup after construction.
            criterion = CrossModalVAE3DebiasedLoss(
                batch_size=args.batch_size,
                temperature=args.temperature,
                lambda_recon=getattr(args, 'lambda_recon', 1.0),
                mask_same_episode=getattr(args, 'debiased_same_episode_mask', True),
                mask_zero_zero=getattr(args, 'debiased_zero_mask', True),
            )
        elif loss_type == 'zero_supcon':
            # SupCon-style: all no-contact frames in a batch form one positive
            # group (in addition to the matched cross-sensor pair). Forces the
            # three encoders to share a single 'no contact' region. Needs
            # per-frame contact labels — train_tri attaches a
            # ContactLabelLookup after construction.
            criterion = CrossModalVAE3ZeroSupConLoss(
                batch_size=args.batch_size,
                temperature=args.temperature,
                lambda_recon=getattr(args, 'lambda_recon', 1.0),
            )
        elif loss_type == 'infonce':
            criterion = CrossModalVAE3Loss(
                batch_size=args.batch_size,
                temperature=args.temperature,
                lambda_recon=getattr(args, 'lambda_recon', 1.0),
            )
        else:
            raise ValueError(
                f"Unknown loss '{loss_type}' for cross_modal_vae3 "
                f"(expected 'infonce', 'uniforce', 'l2', 'cwcl', 'neighbor', "
                f"'debiased', or 'zero_supcon')"
            )
    elif model_name == 'cross_modal_clr3':
        # Tri-modal plain-contrastive baseline: 3-pair NT-Xent, no recon/KL.
        criterion = CrossModalCLR3Loss(
            batch_size=args.batch_size,
            temperature=args.temperature,
        )
    else:
        raise ValueError(f"Model {model_name} not found")
    return criterion

def save_model(model, dataset_name, dataset_type, model_type, dataset_details, model_details, run_name="", tag=None):
    path_out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_type, dataset_name +"_run_" + run_name)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    torch.save(model.state_dict(), os.path.join(path_out, "model.pt"))

    # Periodic weight snapshot (kept, not overwritten) — e.g. tag='ep150'.
    # Only the weights; the yamls below are identical across epochs.
    if tag is not None:
        torch.save(model.state_dict(), os.path.join(path_out, f"model_{tag}.pt"))

    # Save dataset details as yaml file
    dataset_details_path = os.path.join(path_out, "dataset_details.yaml")
    with open(dataset_details_path, 'w') as file:
        yaml.dump(dataset_details, file)

    # Save model details as yaml file
    model_details_path = os.path.join(path_out, "model_details.yaml")
    with open(model_details_path, 'w') as file:
        yaml.dump(model_details, file)

def get_config_details(config_path, config_name):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config[config_name]
    