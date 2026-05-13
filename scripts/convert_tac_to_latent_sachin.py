"""
Convert teleop HDF5 episodes to 16-d latent μ using a trained
cross_modal_vae3 checkpoint (Sachin run with per-pair rot180).

Reads each HDF5 episode, encodes the per-sensor tactile streams to the
shared 16-d latent, and writes a new HDF5 with observation.tac0/tac1
replaced by latents. Everything else is copied verbatim.

Per-sensor I/O schemas (matches the dataset shapes you confirmed):

    Sensor D (Daimon) — vision-based:
        in : observation.image.tac0/tac1   (N, 240, 320, 3) uint8
        ops: (N,H,W,3) → permute (N,3,H,W) → /255 → resize 224 → z-score with
             dataset_details daimon_stats → model._enc(x, 'D')
        out: observation.tac0/tac1         (N, 16) float32

    Sensor E (eFlesh) — magnetic vector:
        in : observation.tac0/tac1         (N, 15) float32
        ops: z-score with eflesh_stats → model._enc(x, 'E')
        out: observation.tac0/tac1         (N, 16) float32

    Sensor F (FlexiTac) — resistive matrix:
        in : observation.tac0/tac1         (N, 192) float32
        ops: reshape (N, 12, 16) → z-score with flexitac_stats →
             model._enc(x, 'F')
        out: observation.tac0/tac1         (N, 16) float32

Optional --inference_transform applies a deterministic fliplr/flipud/rot180
to the raw input before encoding. Useful if the teleop data was collected
in a different orientation than the training data baked into the
dataset_sachin per-pair transforms. Defaults to 'none'.

Usage:
    python scripts/convert_tac_to_latent_sachin.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_<name>_<TS> \
        --sensor E \
        --input_dir  tactile_benchmark/data/recordings/processed/eflesh_eflesh_object_pnp \
        --output_dir tactile_benchmark/data/recordings/processed/eflesh_eflesh_object_pnp_latent
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.utils.models_setup import get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_transform(x, mode):
    """fliplr / flipud / rot180 on the last 1-2 dims (negative indices)."""
    if mode in (None, 'none'):
        return x
    if x.dim() < 2 and mode in ('flipud', 'rot180'):
        return torch.flip(x, dims=[-1]) if mode == 'fliplr' else x
    if mode == 'fliplr':
        return torch.flip(x, dims=[-1])
    if mode == 'flipud':
        return torch.flip(x, dims=[-2])
    if mode == 'rot180':
        return torch.flip(x, dims=[-2, -1])
    raise ValueError(f"Unknown transform: {mode}")


def _zscore(x, mean, std):
    """Channel-wise z-score. mean/std broadcast over leading channel dim."""
    m = torch.tensor(mean, dtype=torch.float32)
    s = torch.tensor(std, dtype=torch.float32)
    # Broadcast: x shape (B, C, H, W) -> reshape stats to (C, 1, 1); (B, 15) -> (15,) is fine.
    if x.dim() == 4 and m.dim() == 1 and m.shape[0] == x.shape[1]:
        m = m.view(1, -1, 1, 1)
        s = s.view(1, -1, 1, 1)
    elif x.dim() == 3 and m.dim() == 1 and m.shape[0] == x.shape[0]:
        # single sample (C, H, W)
        m = m.view(-1, 1, 1)
        s = s.view(-1, 1, 1)
    elif x.dim() == 2 and m.dim() == 1 and m.shape[0] == x.shape[1]:
        # (B, D) — broadcast over batch
        m = m.view(1, -1)
        s = s.view(1, -1)
    return (x - m) / (s + 1e-8)


# ---------------------------------------------------------------------------
# Per-sensor preprocessing
# ---------------------------------------------------------------------------

def preprocess_daimon(raw, dataset_details, image_size, inference_transform='none'):
    """raw: (N, 240, 320, 3) uint8 → (N, 3, image_size, image_size) float32 z-scored."""
    x = torch.from_numpy(raw).float() / 255.0                  # (N, H, W, 3)
    x = x.permute(0, 3, 1, 2).contiguous()                     # (N, 3, H, W)
    x = transforms.functional.resize(x, [image_size, image_size], antialias=True)
    stats = dataset_details.get('daimon_stats', None)
    if stats is not None:
        x = _zscore(x, stats['mean'], stats['std'])
    x = _apply_transform(x, inference_transform)
    return x


def preprocess_eflesh(raw, dataset_details, inference_transform='none'):
    """raw: (N, 15) float32 → (N, 15) z-scored."""
    x = torch.from_numpy(raw).float()                          # (N, 15)
    stats = dataset_details.get('eflesh_stats', None)
    if stats is not None:
        x = _zscore(x, stats['mean'], stats['std'])
    x = _apply_transform(x, inference_transform)               # mostly no-op for 1-D
    return x


def preprocess_flexi(raw, dataset_details, inference_transform='none'):
    """raw: (N, 192) float32 → (N, 12, 16) z-scored."""
    x = torch.from_numpy(raw).float().reshape(-1, 12, 16)      # (N, 12, 16)
    stats = dataset_details.get('flexitac_stats', None)
    if stats is not None:
        # flexi stats are scalar; broadcast to channel-1
        m = float(stats['mean']) if not isinstance(stats['mean'], (list, tuple)) else stats['mean'][0]
        s = float(stats['std']) if not isinstance(stats['std'], (list, tuple)) else stats['std'][0]
        x = (x - m) / (s + 1e-8)
    x = _apply_transform(x, inference_transform)
    return x


# ---------------------------------------------------------------------------
# Encoder forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_batch(model, x, sensor, device, latent_type='mu'):
    """x: (B, ...) on CPU. Returns (B, latent_dim) numpy."""
    x = x.to(device)
    h, mu, logvar, z = model._enc(x, sensor)
    if latent_type == 'mu':
        out = mu
    elif latent_type == 'z':
        out = z
    elif latent_type == 'h':
        out = h
    else:
        raise ValueError(f"Unknown latent_type: {latent_type}")
    return out.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

_SENSOR_TO_HDF5 = {
    'D': ('observation.image.tac0', 'observation.image.tac1'),
    'E': ('observation.tac0', 'observation.tac1'),
    'F': ('observation.tac0', 'observation.tac1'),
}


def process_episode(in_path, out_path, model, sensor, dataset_details,
                    image_size, batch_size, device, latent_type, inference_transform):
    in_key0, in_key1 = _SENSOR_TO_HDF5[sensor]
    with h5py.File(in_path, 'r') as fin, h5py.File(out_path, 'w') as fout:
        # Copy root-level attrs (embodiment, task name, sensor metadata, etc.)
        # The policy data loader reads root.attrs['embodiment'] for per-embodiment
        # norm stats, so this is required.
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        # Copy everything except the keys we're going to replace
        # (observation.tac0/tac1 always; observation.image.tac0/tac1 only when D
        # uses them as input — we keep them around for inspection).
        replace_keys = {'observation.tac0', 'observation.tac1'}
        for key in fin.keys():
            if key in replace_keys:
                continue
            ds_in = fin[key]
            ds_out = fout.create_dataset(key, data=ds_in[:])
            # Preserve per-dataset attrs too (some loaders use these).
            for k, v in ds_in.attrs.items():
                ds_out.attrs[k] = v

        # Encode tac0 and tac1
        for in_key, out_key in (
            (in_key0, 'observation.tac0'),
            (in_key1, 'observation.tac1'),
        ):
            if in_key not in fin:
                tqdm.write(f"  [skip] {in_key} not in {in_path}")
                continue

            raw = fin[in_key][:]                                # numpy
            if sensor == 'D':
                x = preprocess_daimon(raw, dataset_details, image_size, inference_transform)
            elif sensor == 'E':
                x = preprocess_eflesh(raw, dataset_details, inference_transform)
            else:
                x = preprocess_flexi(raw, dataset_details, inference_transform)

            N = x.shape[0]
            outs = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                outs.append(encode_batch(model, x[start:end], sensor, device, latent_type))
            latent = np.concatenate(outs, axis=0)                # (N, latent_dim)
            ds = fout.create_dataset(out_key, data=latent)
            # Copy attrs from the input tac key onto the latent dataset.
            if in_key in fin:
                for k, v in fin[in_key].attrs.items():
                    ds.attrs[k] = v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True,
                        help='checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_...')
    parser.add_argument('--sensor', required=True, choices=['D', 'E', 'F'],
                        help='Which sensor encoder to use.')
    parser.add_argument('--input_dir', required=True,
                        help='Directory of teleop HDF5 episodes.')
    parser.add_argument('--output_dir', required=True,
                        help='Destination for converted HDF5 episodes.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_type', choices=['mu', 'z', 'h'], default='mu')
    parser.add_argument('--inference_transform',
                        choices=['none', 'fliplr', 'flipud', 'rot180'],
                        default='none',
                        help='Optional geometric correction at inference. Apply '
                             'if teleop data orientation differs from the '
                             'training-time canonical (after sachin per-pair rot180).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    with open(os.path.join(args.checkpoint_dir, 'dataset_details.yaml')) as f:
        dataset_details = yaml.safe_load(f)
    image_size = dataset_details.get('image_size', 224)

    model = get_model('cross_modal_vae3', model_details).to(args.device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'model.pt'),
                      map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()

    print('=' * 60)
    print(f"Sensor: {args.sensor}   latent_type: {args.latent_type}   "
          f"inference_transform: {args.inference_transform}")
    print(f"Input : {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print('=' * 60)

    eps = sorted(glob.glob(os.path.join(args.input_dir, 'episode_*.hdf5')))
    if not eps:
        print(f"No episode_*.hdf5 in {args.input_dir}"); return

    for ep_path in tqdm(eps, desc=f"convert[{args.sensor}]"):
        out_path = os.path.join(args.output_dir, os.path.basename(ep_path))
        process_episode(ep_path, out_path, model, args.sensor,
                        dataset_details, image_size, args.batch_size,
                        args.device, args.latent_type, args.inference_transform)
        tqdm.write(f"  → {out_path}")

    print(f"\nDone. {len(eps)} episodes written to {args.output_dir}/")


if __name__ == '__main__':
    main()
