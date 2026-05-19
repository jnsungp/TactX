import os
import torch
from simclr import SimCLR
from simclr.modules import LARS as LARS_simclr
import yaml
from simclr.modules import NT_Xent
from joint_embedding_learning.barlow_twins.barlow import BarlowLoss, LARS
from joint_embedding_learning.utils.losses import CWCL, CrossModalVAELoss, CrossModalVAE3Loss, CrossModalVAE3UniforceLoss, CrossModalVAE3CWCLLoss, CrossModalCLR3Loss


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
    elif model_name == 'barlow_twins':
        criterion = BarlowLoss(args.projector, args.batch_size, args.lambd, args.device)
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
        elif loss_type == 'infonce':
            criterion = CrossModalVAE3Loss(
                batch_size=args.batch_size,
                temperature=args.temperature,
                lambda_recon=getattr(args, 'lambda_recon', 1.0),
            )
        else:
            raise ValueError(
                f"Unknown loss '{loss_type}' for cross_modal_vae3 "
                f"(expected 'infonce', 'uniforce', 'l2', or 'cwcl')"
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
    