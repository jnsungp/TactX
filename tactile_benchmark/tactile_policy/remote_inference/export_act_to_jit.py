#!/usr/bin/env python3
"""
Export ACT checkpoint to TorchScript for deployment.

Usage:
    python export_act_to_jit.py --policy_name pnp_tennis_ball_vis_cs16 --policy_config act_daimon_vis_v2
"""

import dataclasses
import os
import torch
import yaml
from pathlib import Path
import sys

_TACTILE_POLICY_DIR = Path(__file__).resolve().parent.parent
_TACTILE_BENCHMARK_ROOT = _TACTILE_POLICY_DIR.parent
# Repo root must be on path so `import tactile_policy.*` (used inside modeling_act) resolves
sys.path.insert(0, str(_TACTILE_BENCHMARK_ROOT))

import tyro
from tactile_policy.modeling.modeling_act import ACTPolicy

# Repo-relative defaults (override via env if needed)
POLICY_PATH = str(
    Path(os.environ.get("TACTILE_CHECKPOINTS_DIR", str(_TACTILE_BENCHMARK_ROOT / "checkpoints"))).resolve()
) + os.sep
CONFIG_PATH = str(
    Path(os.environ.get("TACTILE_MODEL_CONFIGS_DIR", str(_TACTILE_POLICY_DIR / "configs" / "models"))).resolve()
) + os.sep


class PolicyWrapperTactileImage(torch.nn.Module):
    """Wrapper for tracing ACT policy with tactile image inputs."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image, qpos, tactile_images):
        return self.policy(image, qpos, tactile_images=tactile_images)


class PolicyWrapperTactileArray(torch.nn.Module):
    """Wrapper for tracing ACT policy with tactile array inputs."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image, qpos, tactile_arrays):
        return self.policy(image, qpos, tactile_arrays=tactile_arrays)


class PolicyWrapperNoTactile(torch.nn.Module):
    """Wrapper for tracing ACT policy without tactile inputs."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image, qpos):
        return self.policy(image, qpos)


def export_act_to_jit(checkpoint_path, config_path, output_path):
    """Export ACT checkpoint to TorchScript format."""

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as fp:
        policy_config = yaml.safe_load(fp)

    # Load sensor config if specified
    sensor_config = None
    if 'sensor_config' in policy_config['common'] and policy_config['common']['sensor_config']:
        sensor_config_path = Path(config_path).parent.parent / "sensors" / f"{policy_config['common']['sensor_config']}.yaml"
        print(f"Loading sensor config from: {sensor_config_path}")
        with open(sensor_config_path, "r") as fp:
            sensor_config = yaml.safe_load(fp)

    # Build ACTPolicy config - read dimensions from config (matching main.py)
    state_dim = policy_config['common']['state_dim']
    action_dim = policy_config['common']['action_dim']
    camera_names = policy_config["common"]["camera_names"]
    tactile_names = policy_config["common"].get("tactile_names", [])

    policy_args = {
        'lr': 1e-4,  # Dummy value, not used for inference
        'num_queries': policy_config["model"].get("num_queries", 100),
        'kl_weight': policy_config["model"]["kl_weight"],
        'hidden_dim': policy_config["model"]["hidden_dim"],
        'chunk_size': policy_config["model"].get("chunk_size", 100),
        'dim_feedforward': policy_config["model"]["dim_feedforward"],
        'lr_backbone': float(policy_config["model"]["lr_backbone"]),
        'backbone': policy_config["model"]["backbone"],
        'enc_layers': policy_config["model"]["enc_layers"],
        'dec_layers': policy_config["model"]["dec_layers"],
        'nheads': policy_config["model"]["nheads"],
        'camera_names': camera_names,
        'tactile_names': tactile_names,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'image_feature_strategy': policy_config["model"]["image_feature_strategy"],
        'use_language_conditioning': policy_config["model"]["use_language_conditioning"],
        'gradient_accumulation_steps': 1,
    }

    # Add sensor-specific config if available (matching main.py lines 132-150)
    if sensor_config:
        tactile_modality = sensor_config.get('modality', 'image')
        policy_args['tactile_modality'] = tactile_modality
        policy_args['tactile_dim'] = sensor_config['tactile_dim']
        policy_args['tactile_backbone'] = sensor_config['tactile_backbone']
        policy_args['tactile_image_feature_strategy'] = sensor_config['tactile_image_feature_strategy']
        policy_args['lr_tactile_backbone'] = float(sensor_config['lr_tactile_backbone'])
        if 'tactile_encoding' in sensor_config:
            policy_args['tactile_encoding'] = sensor_config['tactile_encoding']
        if 'tactile_hidden_layers' in sensor_config:
            policy_args['tactile_hidden_layers'] = sensor_config['tactile_hidden_layers']
    else:
        tactile_modality = 'none'
        policy_args['tactile_modality'] = tactile_modality
        policy_args['tactile_dim'] = policy_config['common'].get('tactile_dim', 0)
        policy_args['tactile_backbone'] = policy_config['model'].get('tactile_backbone', 'resnet18')
        policy_args['tactile_image_feature_strategy'] = policy_config['model'].get('tactile_image_feature_strategy', 'ACT_linear')
        policy_args['lr_tactile_backbone'] = float(policy_config['model'].get('lr_tactile_backbone', 1e-4))

    print(f"Creating ACTPolicy with:")
    print(f"  state_dim: {state_dim}")
    print(f"  action_dim: {action_dim}")
    print(f"  cameras: {camera_names}")
    print(f"  tactile_names: {tactile_names}")
    print(f"  tactile_modality: {tactile_modality}")
    print(f"  chunk_size: {policy_args['chunk_size']}")

    # Create policy
    policy = ACTPolicy(policy_args)

    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    policy.load_state_dict(state_dict)

    # Move entire model to CUDA for tracing
    policy = policy.cuda()
    policy.eval()

    print("Checkpoint loaded successfully!")

    # Create example inputs for tracing (on CUDA)
    batch_size = 1
    num_cameras = len(camera_names)
    num_tactile_sensors = len(tactile_names)
    image_height = 240
    image_width = 320

    example_image = torch.randn(batch_size, num_cameras, 3, image_height, image_width).cuda()
    example_qpos = torch.randn(batch_size, state_dim).cuda()

    # Create tactile inputs based on modality
    example_tactile_images = None
    example_tactile_arrays = None

    if tactile_modality == 'image' and num_tactile_sensors > 0:
        # Tactile images have same resolution as camera images
        example_tactile_images = torch.randn(batch_size, num_tactile_sensors, 3, image_height, image_width).cuda()
    elif tactile_modality in ['array', 'state'] and num_tactile_sensors > 0:
        # Tactile arrays: (batch, num_sensors, tactile_dim)
        tactile_dim = policy_args.get('tactile_dim', 0)
        if tactile_dim > 0:
            example_tactile_arrays = torch.randn(batch_size, num_tactile_sensors, tactile_dim).cuda()

    print(f"\nTracing model with example inputs (on CUDA):")
    print(f"  image shape: {example_image.shape}")
    print(f"  qpos shape: {example_qpos.shape}")
    print(f"  tactile_modality: {tactile_modality}")
    if example_tactile_images is not None:
        print(f"  tactile_images shape: {example_tactile_images.shape}")
    if example_tactile_arrays is not None:
        print(f"  tactile_arrays shape: {example_tactile_arrays.shape}")

    # Wrap policy based on tactile modality to avoid None inputs during tracing
    if tactile_modality == 'image' and example_tactile_images is not None:
        wrapped_policy = PolicyWrapperTactileImage(policy)
        trace_inputs = (example_image, example_qpos, example_tactile_images)
    elif tactile_modality in ['array', 'state'] and example_tactile_arrays is not None:
        wrapped_policy = PolicyWrapperTactileArray(policy)
        trace_inputs = (example_image, example_qpos, example_tactile_arrays)
    else:
        wrapped_policy = PolicyWrapperNoTactile(policy)
        trace_inputs = (example_image, example_qpos)

    wrapped_policy.eval()

    # Trace the model (inference mode - no actions/is_pad)
    # Everything must be on CUDA during tracing so embedded constants are CUDA tensors
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_policy,
            trace_inputs,
            check_trace=False  # Disable check to avoid issues with conditional logic
        )

    print("Model traced successfully!")

    # Save traced model
    print(f"\nSaving TorchScript model to: {output_path}")
    traced_model.save(output_path)

    # Verify the saved model can be loaded
    print("\nVerifying saved model...")
    loaded_model = torch.jit.load(output_path)
    with torch.no_grad():
        output = loaded_model(*trace_inputs)

    print(f"✓ Model loaded and tested successfully!")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {policy_args['chunk_size']}, {action_dim})")

    print(f"\n{'='*60}")
    print("Export complete!")
    print(f"{'='*60}")


@dataclasses.dataclass
class Args:
    """Arguments for the export_act_to_jit script."""
    policy_name: str
    policy_config: str


def main(args: Args) -> None:
    checkpoint_path = POLICY_PATH + args.policy_name + "_ckpt/policy_last.ckpt"
    # checkpoint_path = POLICY_PATH + args.policy_name + "_ckpt/policy_iter_90000_seed_0/pytorch_model.bin"
    config_path = CONFIG_PATH + args.policy_config + ".yaml"
    output_path = POLICY_PATH + args.policy_name + "_ckpt/policy_jit.pt"

    export_act_to_jit(checkpoint_path, config_path, output_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
