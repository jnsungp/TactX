import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
from torchvision.utils import make_grid
from joint_embedding_learning.utils.utils import load_optimizer, save_model, get_loss
from joint_embedding_learning.utils.losses import loss_vector_embedding
from joint_embedding_learning.utils.models_setup import get_model


def train_step(args, train_loaders, model, criterion, optimizer, wandb):
    loss_epoch = 0
    loss_vector = 0
    count = 0
    use_supcon = getattr(args, 'use_supcon', False)
    for loader in train_loaders:
        for step, ((x_i, x_j), labels) in tqdm(enumerate(loader)):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            optimizer.zero_grad()
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            if use_supcon:
                episode_ids = labels[1].to(args.device)
                loss = criterion(z_i, z_j, episode_ids)
            else:
                loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_vector_0 = loss_vector_embedding(h_i, h_j)

                if step % 50 == 0:
                    print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")
                    print(f"Step [{step}/{len(loader)}]\t Loss Vector: {loss_vector_0.item()}")

                wandb.log({"Loss_train/Model_Loss_batch": loss.item(), "Loss_train/embedding_diff_batch": loss_vector_0.item(), "Epoch_batch": args.current_epoch})
                args.global_step += 1

                loss_epoch += loss.item()
                loss_vector += loss_vector_0.item()
                count += 1

    loss_epoch /= count
    loss_vector /= count
    return loss_epoch, loss_vector, x_i, x_j


def validation_step(args, val_loaders, model, criterion, wandb, val_name="Val"):
    loss_epoch = 0
    loss_vector = 0
    count = 0
    use_supcon = getattr(args, 'use_supcon', False)

    for val_loader in val_loaders:
        for step, ((x_i, x_j), labels) in tqdm(enumerate(val_loader)):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            h_i, h_j, z_i, z_j = model(x_i, x_j)
            if use_supcon:
                episode_ids = labels[1].to(args.device)
                loss = criterion(z_i, z_j, episode_ids)
            else:
                loss = criterion(z_i, z_j)
            loss_vector_0 = loss_vector_embedding(h_i, h_j)

            if step % 50 == 0:
                print(f"Step [{step}/{len(val_loader)}]\t {val_name} Loss: {loss.item()}")
                print(f"Step [{step}/{len(val_loader)}]\t {val_name} Loss Vector: {loss_vector_0.item()}")

            wandb.log({"Loss_" + val_name + "/Model_Loss_batch": loss.item(), "Loss_" + val_name + "/embedding_diff_batch": loss_vector_0.item(), "Epoch": args.current_epoch})

            loss_epoch += loss.item()
            loss_vector += loss_vector_0.item()
            count += 1

    loss_epoch /= count
    loss_vector /= count
    return loss_epoch, loss_vector, x_i, x_j


def train(train_datasets, validation_datasets, dataset_name, dataset_details, model_details, wandb, run_name='', device='cuda:0'):
    args = argparse.Namespace(device=device, run_name=run_name)

    for key, value in model_details.items():
        setattr(args, key, value)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_type = dataset_details['dataset_type']

    print('Batch Size:', args.batch_size)
    print('Learning Rate:', args.learning_rate)
    print('Temperature:', args.temperature)
    print('Magnetic Input Dim:', args.magnetic_input_dim)
    print('Magnetic Hidden Dims:', args.magnetic_hidden_dims)

    use_episode_sampler = getattr(args, 'use_episode_sampler', False)

    train_loaders = []
    for dataset in train_datasets:
        if use_episode_sampler:
            from joint_embedding_learning.data.samplers import EpisodeBatchSampler
            sampler = EpisodeBatchSampler(
                dataset,
                n_episodes_per_batch=args.n_episodes_per_batch,
                n_frames_per_episode=args.n_frames_per_episode,
            )
            train_loaders.append(torch.utils.data.DataLoader(dataset, batch_sampler=sampler))
            # Override args.batch_size so NT-Xent uses the right size
            args.batch_size = args.n_episodes_per_batch * args.n_frames_per_episode
        else:
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

    model = get_model(model_details)
    model = model.to(args.device)

    optimizer, scheduler = load_optimizer(args, model)
    criterion = get_loss(args)

    args.global_step = 0
    args.current_epoch = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epochs"):
        lr = optimizer.param_groups[0]["lr"]

        loss_epoch, loss_vector, x_i, x_j = train_step(args, train_loaders, model, criterion, optimizer, wandb)
        wandb.log({"Loss_train/Model_Loss": loss_epoch, "Loss_train/embedding_diff": loss_vector, "Misc/learning_rate": lr, "Epoch": epoch})

        if scheduler:
            scheduler.step()

        with torch.no_grad():
            if epoch % 10 == 0:
                save_model(model, dataset_name, dataset_type, 'cross_modal', dataset_details, model_details, run_name=run_name)
                grid = make_grid(x_i[:36], nrow=6, normalize=True)
                wandb.log({"Image_Daimon": [wandb.Image(grid)], "Epoch": epoch})

            for key, val_loader in val_loaders_array.items():
                val_loss_epoch, val_loss_vector, x_i, x_j = validation_step(args, val_loader, model, criterion, wandb, key)
                wandb.log({"Loss_" + key + "/Model_Loss": val_loss_epoch, "Loss_" + key + "/embedding_diff": val_loss_vector, "Epoch": epoch})

            args.current_epoch += 1

    save_model(model, dataset_name, dataset_type, 'cross_modal', dataset_details, model_details, run_name=run_name)
