import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
from tqdm import tqdm
from torchvision.utils import make_grid
from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.utils.utils import load_optimizer, save_model, get_loss
from joint_embedding_learning.utils.losses import loss_vector_embedding
from joint_embedding_learning.utils.models_setup import get_model

from joint_embedding_learning.barlow_twins.barlow import adjust_learning_rate


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

def train_step(args, train_loaders, model_name, model, criterion, optimizer, scaler, wandb):
    loss_epoch = 0
    loss_vector = 0
    count = 0
    loader_len = len(train_loaders[0]) * len(train_loaders)
    step_lr = args.current_epoch * loader_len
    for loader in train_loaders:
        for step, ((x_i, x_j), _) in tqdm(enumerate(loader)):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            if args.optimizer == "Adam":
                # positive pair, with encoding
                # import pdb; pdb.set_trace()
                optimizer.zero_grad()
                h_i, h_j, z_i, z_j = model(x_i, x_j)
                loss = criterion(z_i, z_j)
                loss.backward()
                optimizer.step()

            elif args.optimizer == "LARS":
                adjust_learning_rate(args.epochs, args.batch_size, args.learning_rate_weights, args.learning_rate_biases, optimizer, loader_len, step_lr)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    # positive pair, with encoding
                    h_i, h_j, z_i, z_j = model(x_i, x_j)
                    loss = criterion(z_i, z_j)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                raise NotImplementedError
        
            with torch.no_grad():
                loss_vector_0 = loss_vector_embedding(h_i, h_j)
                m = compute_embedding_metrics(z_i, z_j)

                if step % 50 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")
                    print(f"Step [{step}/{len(loader)}]\t Loss Vector: {loss_vector_0.item()}")

                wandb.log({
                    "Loss_train/Model_Loss_batch": loss.item(),
                    "Loss_train/embedding_diff_batch": loss_vector_0.item(),
                    "Metric_train/pos_cos_mean": m['pos_cos_mean'],
                    "Metric_train/pos_cos_std" : m['pos_cos_std'],
                    "Metric_train/neg_cos_mean": m['neg_cos_mean'],
                    "Metric_train/neg_cos_std" : m['neg_cos_std'],
                    "Metric_train/gap"         : m['gap'],
                    "Metric_train/alignment"   : m['alignment'],
                    "Epoch_batch": args.current_epoch,
                })
                args.global_step += 1

                loss_epoch += loss.item()
                loss_vector += loss_vector_0.item()
                count += 1
                step_lr += 1
    
    loss_epoch /= count
    loss_vector /= count
    return loss_epoch, loss_vector, x_i, x_j

def validation_step(args, val_loaders, model, criterion, wandb, val_name = "Val"):
    loss_epoch = 0
    loss_vector = 0
    count = 0

    for val_loader in val_loaders:
        for step, ((x_i, x_j), _) in tqdm(enumerate(val_loader)):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            loss = criterion(z_i, z_j)
            loss_vector_0 = loss_vector_embedding(h_i, h_j)
            m = compute_embedding_metrics(z_i, z_j)

            if step % 50 == 0:
                print(f"Step [{step}/{len(val_loader)}]\t" + val_name + "Loss: {loss.item()}")
                print(f"Step [{step}/{len(val_loader)}]\t" + val_name + "Loss Vector: {loss_vector_0.item()}")

            wandb.log({
                "Loss_" + val_name + "/Model_Loss_batch": loss.item(),
                "Loss_" + val_name + "/embedding_diff_batch": loss_vector_0.item(),
                f"Metric_{val_name}/pos_cos_mean": m['pos_cos_mean'],
                f"Metric_{val_name}/pos_cos_std" : m['pos_cos_std'],
                f"Metric_{val_name}/neg_cos_mean": m['neg_cos_mean'],
                f"Metric_{val_name}/neg_cos_std" : m['neg_cos_std'],
                f"Metric_{val_name}/gap"         : m['gap'],
                f"Metric_{val_name}/alignment"   : m['alignment'],
                "Epoch": args.current_epoch,
            })

            loss_epoch += loss.item()
            loss_vector += loss_vector_0.item()
            count += 1

    loss_epoch /= count
    loss_vector /= count
    return loss_epoch, loss_vector, x_i, x_j


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
    
    train_loaders = []
    for dataset in train_datasets:
        train_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True))
    
    val_loaders_array = {}
    for key, value in validation_datasets.items():
        val_loaders = []
        for dataset in value:
            if len(dataset) == 0:
                continue
            val_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True))
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
                if model_name == 'cross_modal':
                    grid = make_grid(x_i[:36], nrow=6, normalize=True)
                    wandb.log({"Image_Daimon": [wandb.Image(grid)], "Epoch": epoch})
                else:
                    grid_i = make_grid(x_i[:36], nrow=6, normalize=True)
                    grid_j = make_grid(x_j[:36], nrow=6, normalize=True)
                    wandb.log({"Image": [wandb.Image(grid_i), wandb.Image(grid_j)], "Epoch": epoch})

            for key, val_loader in val_loaders_array.items():
                val_loss_epoch, val_loss_vector, x_i, x_j = validation_step(args, val_loader, model, criterion, wandb, key)
                wandb.log({"Loss_" + key + "/Model_Loss": val_loss_epoch, "Loss_" + key + "/embedding_diff": val_loss_vector, "Epoch": epoch})

            args.current_epoch += 1

    save_model(model, dataset_name, dataset_type, model_type, dataset_details, model_details, run_name=run_name)