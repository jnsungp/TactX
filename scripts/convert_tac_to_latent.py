"""
Convert observation.tac0 / observation.tac1 in HDF5 episodes to latent space z (64-dim)
using a trained CrossModalCLR checkpoint.

  - daimon: tac0/tac1 are (N, 384000) raw sensor → extract depth (240×320) → image encoder → z

Usage:
  python scripts/convert_tac_to_latent.py
  python scripts/convert_tac_to_latent.py --device cuda:1
  python scripts/convert_tac_to_latent.py --batch_size 128
"""

import os
import sys
import argparse
import glob

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.utils.models_setup import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Convert tac0/tac1 to latent z using trained CrossModalCLR")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints/cross_modal/cross_modal/dataset_cross_modal_run_20260411_220509",
                        help="Path to checkpoint directory containing model.pt and yaml configs")
    parser.add_argument("--daimon_input_dir", type=str,
                        default="/home/junsung/cross_modal_tactile/dai_dai_small_ball/processed",
                        help="Directory with daimon HDF5 episodes (tac0/tac1 shape: N×384000)")
    parser.add_argument("--daimon_output_dir", type=str,
                        default="/home/junsung/cross_modal_tactile/dai_dai_small_ball/latent",
                        help="Output directory for daimon latent episodes")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def load_model(checkpoint_dir, device):
    """Load trained CrossModalCLR model from checkpoint."""
    model_details_path = os.path.join(checkpoint_dir, "model_details.yaml")
    dataset_details_path = os.path.join(checkpoint_dir, "dataset_details.yaml")
    model_pt_path = os.path.join(checkpoint_dir, "model.pt")

    with open(model_details_path, 'r') as f:
        model_details = yaml.safe_load(f)
    with open(dataset_details_path, 'r') as f:
        dataset_details = yaml.safe_load(f)

    model = get_model('cross_modal', model_details)
    state_dict = torch.load(model_pt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, model_details, dataset_details


def build_daimon_transform(dataset_details):
    """Build image transforms matching the training pipeline for Daimon depth."""
    image_size = dataset_details.get('image_size', 224)
    stats = dataset_details.get('daimon_stats', {})
    mean = stats.get('mean', [0.0, 0.0, 0.0])
    std = stats.get('std', [1.0, 1.0, 1.0])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(torch.tensor(mean), torch.tensor(std)),
    ])


def extract_depth_from_tac(raw_frame):
    """Extract depth from raw daimon tac data (384000,) → (240, 320)."""
    H, W = 240, 320
    return raw_frame[:H * W].reshape(H, W)


@torch.no_grad()
def encode_daimon_batch(model, batch, transform, device):
    """
    Encode a batch of daimon depth frames to latent z.
    batch: (B, 240, 320) numpy float32
    Returns: (B, 64) numpy float32
    """
    tensors = []
    for frame in batch:
        t = torch.from_numpy(frame).float().unsqueeze(0)  # (1, 240, 320)
        t = t.repeat(3, 1, 1)                              # (3, 240, 320)
        t = transform(t)                                    # (3, 224, 224)
        tensors.append(t)

    x = torch.stack(tensors, dim=0).to(device)  # (B, 3, 224, 224)
    h = model.image_encoder(x)                  # (B, 512)
    z = model.projector_image(h)                # (B, 64)
    return z.cpu().numpy()


def process_episodes(input_dir, output_dir, model, dataset_details, batch_size, device):
    """
    Process all HDF5 episodes: convert tac0/tac1 → latent z, copy everything else.
    """
    os.makedirs(output_dir, exist_ok=True)

    episode_files = sorted(glob.glob(os.path.join(input_dir, "episode_*.hdf5")))
    if not episode_files:
        print(f"  No episodes found in {input_dir}")
        return

    transform = build_daimon_transform(dataset_details)

    for ep_path in tqdm(episode_files, desc="  daimon"):
        ep_name = os.path.basename(ep_path)
        out_path = os.path.join(output_dir, ep_name)

        with h5py.File(ep_path, 'r') as fin, h5py.File(out_path, 'w') as fout:
            for key in fin.keys():
                if key in ('observation.tac0', 'observation.tac1'):
                    continue
                data = fin[key][:]
                fout.create_dataset(key, data=data)

            for tac_key in ('observation.tac0', 'observation.tac1'):
                if tac_key not in fin:
                    continue

                raw_data = fin[tac_key][:]  # (N, 384000)
                N = raw_data.shape[0]
                latent_all = np.zeros((N, 64), dtype=np.float32)

                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    batch_raw = raw_data[start:end]
                    depths = np.stack([extract_depth_from_tac(frame) for frame in batch_raw])
                    latent = encode_daimon_batch(model, depths, transform, device)
                    latent_all[start:end] = latent

                fout.create_dataset(tac_key, data=latent_all)

        tqdm.write(f"    {ep_name}: {tac_key} {raw_data.shape} → (N, 64)")


def main():
    args = parse_args()

    checkpoint_dir = args.checkpoint_dir
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            checkpoint_dir
        )

    print("=" * 60)
    print("Convert tac0/tac1 → Latent Space z (64-dim)")
    print("=" * 60)
    print(f"  Checkpoint : {checkpoint_dir}")
    print(f"  Device     : {args.device}")
    print(f"  Batch size : {args.batch_size}")
    print()

    model, model_details, dataset_details = load_model(checkpoint_dir, args.device)
    print(f"  Model loaded: projection_dim={model_details['projection_dim']}, "
          f"resnet={model_details['resnet']}")
    print()

    # --- Daimon dataset (384000-dim → depth image → image_encoder → z) ---
    if os.path.isdir(args.daimon_input_dir):
        print(f"[Daimon] {args.daimon_input_dir}")
        print(f"         → {args.daimon_output_dir}")
        process_episodes(
            args.daimon_input_dir, args.daimon_output_dir,
            model, dataset_details, args.batch_size, args.device
        )
        print()
    else:
        print(f"[Daimon] SKIPPED — {args.daimon_input_dir} not found")
        print()

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
