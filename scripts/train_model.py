from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.training import train, train_tri, train_tri_dynamic_pretrain
import torch
import os
import yaml
import wandb
from datetime import datetime

if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='CMTJE')
    parser.add_argument('--dataset_name', type=str, default='dataset_2', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='simclr', help='Name of the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--run_name', type=str, default='', help='Name of the run')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--objects', nargs='+', default=None,
                        help='Object names to train on (e.g. --objects cylinder plug). '
                             'Requires per-object data layout. If omitted, all objects are used.')
    parser.add_argument('--phase', choices=['dynamic_pretrain', 'paired'], default='paired',
                        help='cross_modal_vae3: dynamic_pretrain (Stage 1, self-recon only) '
                             'or paired (Stage 2 / default, full pair training).')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Optional path to a Stage 1 checkpoint to load weights from '
                             'before Stage 2 paired training.')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.run_name:
        args.run_name = f"{args.run_name}_{timestamp}"
    else:
        args.run_name = timestamp

    # Dataset configuration
    dataset_config =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'datasets_config.yaml')
    with open(dataset_config, 'r') as file:
        config = yaml.safe_load(file)
    dataset_details = config[args.dataset_name]

    # Model configuration
    model_config =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'models_config.yaml')
    with open(model_config, 'r') as file:
        config = yaml.safe_load(file)
    model_details = config[args.model_name]

    if args.debug:
        project = 'CMTEJ_debug'
    else:
        project = 'CMTEJ_comparison'
    output_folder = f"{args.model_name}_{dataset_details['dataset_type']}_{args.dataset_name}_{args.run_name}"

    wandb.init(
            project=project,
            name=output_folder,
            id=output_folder,
            resume=False,
        )
    
    print(f"Run name: {args.run_name}")
    print(f"Checkpoint will be saved to: checkpoints/{dataset_details['dataset_type']}/{args.model_name}/{args.dataset_name}_run_{args.run_name}/")

    # Dataset loading
    train_datasets, val_unseen_grasps, val_unseen_tools = get_all_contrastive_datasets(
        args.dataset_name, device=args.device, objects=args.objects,
    )

    # Model Training (dispatch on tri-modal vs. legacy pair model)
    if args.model_name == 'cross_modal_vae3':
        if args.phase == 'dynamic_pretrain':
            # Stage 1: dynamic self-recon pretrain (unpaired). train_datasets is
            # {'D': ds, 'E': ds, 'F': ds}; val_unseen_grasps is the matching val dict.
            train_tri_dynamic_pretrain(
                train_datasets, val_unseen_grasps, args.dataset_name,
                dataset_details, args.model_name, model_details, wandb,
                run_name=args.run_name, device=args.device,
            )
        else:
            # Stage 2 (default): paired full training. train_datasets is
            # {'de': ds, 'ef': ds, 'fd': ds}; val_unseen_grasps is matching val dict.
            validation_datasets = {'val': val_unseen_grasps}
            train_tri(
                train_datasets, validation_datasets, args.dataset_name,
                dataset_details, args.model_name, model_details, wandb,
                run_name=args.run_name, device=args.device,
                resume_from=args.resume_from,
            )
    else:
        validation_datasets = {
            'val_unseen_grasps': val_unseen_grasps,
            'val_unseen_tools': val_unseen_tools
        }
        train(train_datasets, validation_datasets, args.dataset_name, dataset_details, args.model_name, model_details, wandb, run_name=args.run_name, device=args.device)