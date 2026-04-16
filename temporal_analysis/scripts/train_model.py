from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.training import train
import os
import yaml
import wandb
from datetime import datetime

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Cross-Modal Tactile Training')
    parser.add_argument('--dataset_name', type=str, default='dataset_cross_modal', help='Name of the dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--run_name', type=str, default='', help='Name of the run')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--projection_dim', type=int, default=None, help='Override projection/latent dim from config')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate from config')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    # Temporal sampling options
    parser.add_argument('--thin_hold_factor', type=int, default=None,
                        help='Subsample hold-phase frames by this factor. e.g. 5 = keep 1 in 5 hold frames.')
    parser.add_argument('--thin_threshold_ratio', type=float, default=1.5,
                        help='Motion threshold = median(motion) * this ratio. Higher = more aggressive thinning.')
    parser.add_argument('--use_episode_sampler', action='store_true',
                        help='Use EpisodeBatchSampler (groups by episode). Overrides --batch_size.')
    parser.add_argument('--use_supcon', action='store_true',
                        help='Use SupCon loss: same-episode frames are positives, cross-episode are negatives. '
                             'Requires --use_episode_sampler to have multiple frames per episode per batch.')
    parser.add_argument('--n_episodes_per_batch', type=int, default=16,
                        help='Episodes per batch when --use_episode_sampler.')
    parser.add_argument('--n_frames_per_episode', type=int, default=16,
                        help='Frames per episode when --use_episode_sampler.')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.run_name:
        args.run_name = f"{args.run_name}_{timestamp}"
    else:
        args.run_name = timestamp

    # Dataset configuration
    dataset_config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'datasets_config.yaml')
    with open(dataset_config, 'r') as file:
        config = yaml.safe_load(file)
    dataset_details = config[args.dataset_name]

    # Model configuration
    model_config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'models_config.yaml')
    with open(model_config, 'r') as file:
        config = yaml.safe_load(file)
    model_details = config['cross_modal']

    # CLI overrides for model hyperparams
    if args.batch_size is not None:
        model_details['batch_size'] = args.batch_size
    if args.projection_dim is not None:
        model_details['projection_dim'] = args.projection_dim
    if args.epochs is not None:
        model_details['epochs'] = args.epochs
    if args.lr is not None:
        model_details['learning_rate'] = args.lr

    # Temporal sampling options passed through model_details so training.py sees them via args
    if args.use_episode_sampler:
        model_details['use_episode_sampler'] = True
        model_details['n_episodes_per_batch'] = args.n_episodes_per_batch
        model_details['n_frames_per_episode'] = args.n_frames_per_episode
        # batch_size will be overridden inside train() to n_eps * n_frames
        model_details['batch_size'] = args.n_episodes_per_batch * args.n_frames_per_episode

    if args.use_supcon:
        model_details['use_supcon'] = True
        if not args.use_episode_sampler:
            print("WARNING: --use_supcon without --use_episode_sampler means batches will "
                  "have mostly single-frame-per-episode samples, defeating SupCon's purpose.")

    project = 'CMTEJ_debug' if args.debug else 'CMTEJ_comparison'
    output_folder = f"cross_modal_{dataset_details['dataset_type']}_{args.dataset_name}_{args.run_name}"

    wandb.init(
        project=project,
        name=output_folder,
        id=output_folder,
        resume=False,
    )

    print(f"Run name: {args.run_name}")
    print(f"Checkpoint will be saved to: checkpoints/{dataset_details['dataset_type']}/cross_modal/{args.dataset_name}_run_{args.run_name}/")
    if args.thin_hold_factor:
        print(f"Thinning: hold_factor={args.thin_hold_factor}, threshold_ratio={args.thin_threshold_ratio}")
    if args.use_episode_sampler:
        print(f"Sampler: {args.n_episodes_per_batch} eps x {args.n_frames_per_episode} frames = "
              f"{args.n_episodes_per_batch * args.n_frames_per_episode} batch size")

    # Dataset loading (thinning applies to train only)
    train_datasets, val_unseen_grasps, val_unseen_tools = get_all_contrastive_datasets(
        args.dataset_name,
        device=args.device,
        thin_hold_factor=args.thin_hold_factor,
        thin_threshold_ratio=args.thin_threshold_ratio,
    )

    validation_datasets = {
        'val_unseen_grasps': val_unseen_grasps,
        'val_unseen_tools': val_unseen_tools
    }

    # Model Training
    train(train_datasets, validation_datasets, args.dataset_name, dataset_details, model_details, wandb, run_name=args.run_name, device=args.device)
