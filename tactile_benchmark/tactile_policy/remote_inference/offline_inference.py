#!/usr/bin/env python3
"""
Offline inference for ACT policy using checkpoint directly (no TorchScript needed).

Usage:
    python tactile_policy/remote_inference/offline_inference.py \
        --checkpoint_path /media/ExtHDD/anya/tactile_proj/checkpoints/pnp_force_ckpt/policy_last.ckpt \
        --config_path /home/anya/Workspace/tactile_benchmark/tactile_policy/configs/models/act_force.yaml \
        --norm_stats_path /media/ExtHDD/anya/tactile_proj/checkpoints/pnp_force_ckpt/dataset_stats.pkl \
        --test_hdf_path data/recordings/processed/pnp_flexi_val/episode_0000.hdf5 \
        --chunk_size 100
"""

import torch
import numpy as np
import h5py
import yaml
import pickle
import argparse
import cv2
import sys
from pathlib import Path
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tactile_policy.modeling.modeling_act import ACTPolicy
from tactile_policy.modeling.utils import make_visual_encoder


def load_and_preprocess_image(img_data):
    """Load and preprocess image data (handles both compressed and raw formats)."""
    if len(img_data.shape) == 1:  # Compressed JPEG
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
    elif len(img_data.shape) == 3:
        if img_data.shape[-1] == 3:  # HWC format
            img = img_data.transpose(2, 0, 1)  # HWC -> CHW
        else:  # Already CHW
            img = img_data
    else:
        raise ValueError(f"Unexpected image shape: {img_data.shape}")
    return img


def load_policy(checkpoint_path, config_path):
    """Load ACT policy from checkpoint."""
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build policy config (must match training config)
    policy_config = {
        'lr': 1e-4,  # Dummy, not used for inference
        'num_queries': config['model'].get('num_queries', 100),
        'kl_weight': config['model']['kl_weight'],
        'hidden_dim': config['model']['hidden_dim'],
        'chunk_size': config['model'].get('chunk_size', 100),
        'dim_feedforward': config['model']['dim_feedforward'],
        'lr_backbone': float(config['model']['lr_backbone']),
        'backbone': config['model']['backbone'],
        'enc_layers': config['model']['enc_layers'],
        'dec_layers': config['model']['dec_layers'],
        'nheads': config['model']['nheads'],
        'camera_names': config['common']['camera_names'],
        'tactile_names': config['common'].get('tactile_names', []),
        'state_dim': config['common']['state_dim'],
        'action_dim': config['common']['action_dim'],
        'image_feature_strategy': config['model']['image_feature_strategy'],
        'use_language_conditioning': config['model']['use_language_conditioning'],
        'gradient_accumulation_steps': 1,
        'tactile_modality': 'none',  # Set based on sensor config if needed
        'tactile_dim': 0,
    }
    
    # Load sensor config if specified
    if 'sensor_config' in config['common'] and config['common']['sensor_config']:
        sensor_config_path = Path(config_path).parent.parent / 'sensors' / f"{config['common']['sensor_config']}.yaml"
        if sensor_config_path.exists():
            with open(sensor_config_path, 'r') as f:
                sensor_config = yaml.safe_load(f)
            
            # Add all sensor-specific config (matching main.py)
            policy_config['tactile_modality'] = sensor_config.get('modality', 'image')
            policy_config['tactile_dim'] = sensor_config['tactile_dim']
            policy_config['tactile_backbone'] = sensor_config['tactile_backbone']
            policy_config['tactile_image_feature_strategy'] = sensor_config['tactile_image_feature_strategy']
            policy_config['lr_tactile_backbone'] = float(sensor_config['lr_tactile_backbone'])
            if 'tactile_encoding' in sensor_config:
                policy_config['tactile_encoding'] = sensor_config['tactile_encoding']
            if 'tactile_hidden_layers' in sensor_config:
                policy_config['tactile_hidden_layers'] = sensor_config['tactile_hidden_layers']
            
            print(f"Loaded sensor config: {sensor_config_path}")
            print(f"  Tactile modality: {policy_config['tactile_modality']}")
            print(f"  Tactile dim: {policy_config['tactile_dim']}")
            print(f"  Tactile backbone: {policy_config['tactile_backbone']}")
    else:
        # Default values when no sensor config (matching main.py)
        policy_config['tactile_modality'] = 'none'
        policy_config['tactile_dim'] = config['common'].get('tactile_dim', 0)
        policy_config['tactile_backbone'] = config['model'].get('tactile_backbone', 'resnet18')
        policy_config['tactile_image_feature_strategy'] = config['model'].get('tactile_image_feature_strategy', 'ACT_linear')
        policy_config['lr_tactile_backbone'] = float(config['model'].get('lr_tactile_backbone', 1e-4))
    
    print(f"\nCreating ACTPolicy with:")
    print(f"  state_dim: {policy_config['state_dim']}")
    print(f"  action_dim: {policy_config['action_dim']}")
    print(f"  cameras: {policy_config['camera_names']}")
    print(f"  chunk_size: {policy_config['chunk_size']}")
    
    # Create visual encoder and policy
    visual_encoder, visual_preprocessor = make_visual_encoder('ACT', policy_config)
    policy = ACTPolicy(policy_config)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    policy.load_state_dict(state_dict)
    
    policy.eval()
    print("✓ Policy loaded successfully!")
    
    return policy, visual_preprocessor, policy_config


def run_inference(policy, visual_preprocessor, norm_stats, test_hdf_path, chunk_size, device='cuda', policy_config=None):
    """Run inference on test episode."""
    print(f"\n{'='*60}")
    print(f"Running inference on: {test_hdf_path}")
    print(f"{'='*60}")
    
    policy = policy.to(device)
    
    with h5py.File(test_hdf_path, 'r') as f:
        num_timesteps = f['observation.state'].shape[0]
        print(f"Episode length: {num_timesteps} timesteps")
        
        # Get embodiment for normalization
        embodiment = f.attrs.get('embodiment', 'franka')
        stats = norm_stats[embodiment]
        print(f"Using normalization stats for embodiment: {embodiment}")
        
        # Check if tactile data is needed
        tactile_names = policy_config.get('tactile_names', []) if policy_config else []
        tactile_modality = policy_config.get('tactile_modality', 'none') if policy_config else 'none'
        use_tactile = tactile_modality in ['array', 'state', 'force'] and tactile_names
        
        if use_tactile:
            print(f"Using tactile data: {tactile_names} (modality: {tactile_modality})")
        
        all_predictions = []
        all_ground_truth = []
        
        # Run inference for first 10 timesteps (or full episode)
        num_samples = num_timesteps
        print(f"\nRunning inference for {num_samples} timesteps...")
        
        for t in range(num_samples):
            # Load observation at timestep t
            cam0 = f['observation.image.cam0'][t]
            cam1 = f['observation.image.cam1'][t]
            qpos = f['observation.state'][t]
            action_gt = f['action'][t]
            
            # Preprocess images
            cam0_img = load_and_preprocess_image(cam0)
            cam1_img = load_and_preprocess_image(cam1)
            images = np.stack([cam0_img, cam1_img], axis=0)  # [2, 3, H, W]
            
            # Normalize qpos
            qpos_normalized = (qpos - stats['qpos_mean']) / (stats['qpos_std'] + 1e-6)
            
            # Prepare inputs
            images_tensor = visual_preprocessor(images[None]).to(device)  # [1, 2, 3, 240, 320]
            qpos_tensor = torch.from_numpy(qpos_normalized[None]).float().to(device)  # [1, 128]
            
            # Load and normalize tactile data if needed
            tactile_arrays_tensor = None
            if use_tactile:
                tactile_list = []
                for tactile_name in tactile_names:
                    tactile_key = f'observation.{tactile_name}'
                    if tactile_key in f:
                        tactile_data = f[tactile_key][t]
                        tactile_data = np.atleast_1d(tactile_data)
                        
                        # Check for inf/nan
                        if np.any(~np.isfinite(tactile_data)):
                            tactile_data = np.nan_to_num(tactile_data, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        tactile_list.append(tactile_data)
                
                if tactile_list:
                    # Concatenate all tactile sensors
                    tactile_concat = np.concatenate(tactile_list, axis=0)  # (num_tactile * tactile_dim,)
                    
                    # Normalize if stats available
                    if 'tactile_mean' in stats and 'tactile_std' in stats:
                        tactile_concat = (tactile_concat - stats['tactile_mean']) / (stats['tactile_std'] + 1e-6)
                    
                    # Reshape to (num_tactile, tactile_dim)
                    num_tactile = len(tactile_names)
                    tactile_dim = len(tactile_concat) // num_tactile
                    tactile_reshaped = tactile_concat.reshape(num_tactile, tactile_dim)
                    
                    # Convert to tensor: [1, num_tactile, tactile_dim]
                    tactile_arrays_tensor = torch.from_numpy(tactile_reshaped[None]).float().to(device)
            
            # Inference
            with torch.no_grad():
                if use_tactile and tactile_arrays_tensor is not None:
                    actions = policy(images_tensor, qpos_tensor, tactile_arrays=tactile_arrays_tensor)
                else:
                    actions = policy(images_tensor, qpos_tensor)  # [1, chunk_size, 128]
            
            # Denormalize actions
            actions_np = actions[0].cpu().numpy()  # [chunk_size, 128]
            actions_denorm = actions_np * stats['action_std'] + stats['action_mean']
            
            all_predictions.append(actions_denorm[0])  # First action in chunk
            all_ground_truth.append(action_gt)
            
            if t < 3:  # Print first 3 timesteps
                print(f"\nTimestep {t}:")
                print(f"  Predicted action (first 8 dims): {actions_denorm[0, :8]}")
                print(f"  Ground truth action (first 8 dims): {action_gt[:8]}")
                print(f"  Difference: {np.abs(actions_denorm[0, :8] - action_gt[:8])}")
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_ground_truth = np.array(all_ground_truth)
        
        mae = np.mean(np.abs(all_predictions - all_ground_truth))
        mse = np.mean((all_predictions - all_ground_truth) ** 2)

        # Use only first 8 dims (real action)
        pred8 = all_predictions[:, :8]
        gt8   = all_ground_truth[:, :8]

        mae8 = np.mean(np.abs(pred8 - gt8))
        mse8 = np.mean((pred8 - gt8) ** 2)
        
        print(f"\n{'='*60}")
        print(f"Inference Results:")
        print(f"{'='*60}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Per-dimension MAE (first 8 dims): {mae8:.6f}")
        print(f"Per-dimension MSE (first 8 dims): {mse8:.6f}")

        print(f"\nPer-dimension MAE (first 8 dims):")
        for i in range(min(8, all_predictions.shape[1])):
            dim_mae = np.mean(np.abs(all_predictions[:, i] - all_ground_truth[:, i]))
            print(f"  Dim {i}: {dim_mae:.6f}")

        return {
            "mae": float(mae),
            "mse": float(mse),
            "mae8": float(mae8),
            "mse8": float(mse8),
        }


def resolve_test_paths(test_hdf_path):
    """Resolve a single path, directory, or glob into a sorted list of HDF5 files."""
    path = Path(test_hdf_path)
    if path.is_dir():
        candidates = sorted(path.glob("*.hdf5"))
    elif any(ch in test_hdf_path for ch in ["*", "?", "["]):
        candidates = sorted(Path(p) for p in glob.glob(test_hdf_path))
    else:
        candidates = [path]
    return [p for p in candidates if p.is_file()]

def main():
    parser = argparse.ArgumentParser(description="Offline inference for ACT policy")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint file (.ckpt)")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to model config YAML")
    parser.add_argument("--norm_stats_path", type=str, required=True,
                       help="Path to dataset_stats.pkl")
    parser.add_argument("--test_hdf_path", type=str, required=True,
                       help="Path to test HDF5 episode")
    parser.add_argument("--chunk_size", type=int, default=100,
                       help="Action chunk size")
    parser.add_argument("--device", type=str, default='cuda',
                       help="Device to run inference on (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load normalization stats
    print(f"Loading normalization stats from: {args.norm_stats_path}")
    with open(args.norm_stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    print(f"✓ Loaded stats for embodiments: {list(norm_stats.keys())}")
    
    # Load policy
    policy, visual_preprocessor, policy_config = load_policy(
        args.checkpoint_path, 
        args.config_path
    )
    
    # Run inference (supports single file, directory, or glob)
    test_paths = resolve_test_paths(args.test_hdf_path)
    if not test_paths:
        raise FileNotFoundError(f"No HDF5 files found for: {args.test_hdf_path}")

    metrics_list = []
    for idx, test_path in enumerate(test_paths):
        print(f"\n>>> [{idx + 1}/{len(test_paths)}] {test_path}")
        metrics = run_inference(
            policy,
            visual_preprocessor,
            norm_stats,
            str(test_path),
            args.chunk_size,
            args.device,
            policy_config,
        )
        metrics_list.append(metrics)

    if len(metrics_list) > 1:
        avg_metrics = {
            "mae": float(np.mean([m["mae"] for m in metrics_list])),
            "mse": float(np.mean([m["mse"] for m in metrics_list])),
            "mae8": float(np.mean([m["mae8"] for m in metrics_list])),
            "mse8": float(np.mean([m["mse8"] for m in metrics_list])),
        }
        print(f"\n{'='*60}")
        print(f"Average Results ({len(metrics_list)} episodes):")
        print(f"{'='*60}")
        print(f"Mean Absolute Error (MAE): {avg_metrics['mae']:.6f}")
        print(f"Mean Squared Error (MSE): {avg_metrics['mse']:.6f}")
        print(f"Per-dimension MAE (first 8 dims): {avg_metrics['mae8']:.6f}")
        print(f"Per-dimension MSE (first 8 dims): {avg_metrics['mse8']:.6f}")


if __name__ == "__main__":
    main()