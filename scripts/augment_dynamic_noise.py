"""
Augment paired dynamic data by adding Gaussian noise.

Reads all .pt files from each pair's sensor directories under --src,
copies the originals and generates N augmented copies with Gaussian noise.
The noise std is scaled relative to each sensor's data_std from the training
config (dataset_sachin_static).

Pair consistency: within a pair directory (e.g. de_pair), both sensors get
augmented with the same aug_id and frame indexing. Each sensor gets
independent noise draws (since they have different physical units/scales).

Output:
    --dst/{de,ef,fd}_pair/{sensor}/pnp/ep####_fr####.pt
    Originals: ep0000-ep0004 (copied verbatim)
    Augmented: ep0005-ep0049 (9 copies x 5 originals)

Usage:
    python scripts/augment_dynamic_noise.py \
        --src latent_data_processed_dynamic/pnp \
        --dst latent_data_processed_dynamic/pnp_aug \
        --n_copies 9 --noise_scale 0.1
"""

import argparse
import glob
import os
import re
import shutil

import torch

# Sensor data std from dataset_sachin_static (raw scale, pre z-score).
# Used to scale the noise magnitude appropriately per sensor.
_SENSOR_STD = {
    'daimon': torch.tensor([0.219750, 0.102873, 0.097109]),
    'eflesh': torch.tensor([
        680.086060, 1364.270508, 517.791931, 1804.439331, 484.050659,
        1545.929688, 595.985901, 2031.554932, 1672.495728, 592.037659,
        2192.061768, 738.748596, 2003.141113, 318.000000, 530.152588,
    ]),
    'flexi': torch.tensor([0.182583]),
}

_EP_FR_RE = re.compile(r'ep(\d+)_fr(\d+)\.pt$')

# Pair definitions: pair_name -> (sensor_a_folder, sensor_b_folder)
_PAIRS = {
    'de_pair': ('daimon', 'eflesh'),
    'ef_pair': ('eflesh', 'flexi'),
    'fd_pair': ('flexi', 'daimon'),
}


def _parse_ep_fr(filename):
    m = _EP_FR_RE.search(filename)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _get_noise_std_for_sensor(sensor_name, noise_scale):
    """Return a broadcastable std tensor for the given sensor."""
    base_std = _SENSOR_STD[sensor_name]
    return base_std * noise_scale


def _add_noise(x, sensor_name, noise_scale, rng):
    """Add scaled Gaussian noise to tensor x."""
    std = _get_noise_std_for_sensor(sensor_name, noise_scale)
    # Broadcast std to match x shape
    if x.dim() == 3 and sensor_name == 'daimon':
        # (3, H, W) -> std shape (3,) -> (3, 1, 1)
        std = std.view(3, 1, 1)
    elif x.dim() == 1:
        # (15,) for eflesh or (192,) for flexi
        if std.numel() == 1:
            pass  # scalar broadcasts naturally
        # else std is already (15,) matching x
    elif x.dim() == 2 and sensor_name == 'flexi':
        # (12, 16) -> scalar std
        pass  # scalar broadcasts
    noise = torch.randn(x.shape, generator=rng) * std
    return x + noise


def _load_raw(path):
    """Load a .pt file and return the raw tensor (unwrap dict if needed)."""
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(raw, dict):
        if   'image'    in raw: return raw['image'].float(), 'image'
        elif 'magnetic' in raw: return raw['magnetic'].float(), 'magnetic'
        elif 'flexi'    in raw: return raw['flexi'].float(), 'flexi'
        elif 'flexitac' in raw: return raw['flexitac'].float(), 'flexitac'
        else:
            raise ValueError(f"Unknown keys {list(raw.keys())} in {path}")
    return raw.float(), None


def _save_raw(tensor, key, path):
    """Save tensor, wrapping in dict if original was dict-wrapped."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if key is not None:
        torch.save({key: tensor}, path)
    else:
        torch.save(tensor, path)


def process_pair(pair_name, src_root, dst_root, n_copies, noise_scale, seed):
    """Process one pair directory: copy originals + generate augmented copies."""
    sensor_a, sensor_b = _PAIRS[pair_name]
    src_pair = os.path.join(src_root, pair_name)

    if not os.path.isdir(src_pair):
        print(f"  [skip] {src_pair} not found")
        return

    # Find the object subdirectory (e.g. 'pnp')
    # Sensor dirs contain object sub-dirs with .pt files
    sensor_a_dir = os.path.join(src_pair, sensor_a)
    sensor_b_dir = os.path.join(src_pair, sensor_b)

    # Discover object sub-dirs
    obj_dirs_a = [d for d in os.listdir(sensor_a_dir)
                  if os.path.isdir(os.path.join(sensor_a_dir, d))]

    for obj_name in obj_dirs_a:
        src_a = os.path.join(sensor_a_dir, obj_name)
        src_b = os.path.join(sensor_b_dir, obj_name)
        dst_a = os.path.join(dst_root, pair_name, sensor_a, obj_name)
        dst_b = os.path.join(dst_root, pair_name, sensor_b, obj_name)
        os.makedirs(dst_a, exist_ok=True)
        os.makedirs(dst_b, exist_ok=True)

        # List all .pt files for sensor_a, grouped by episode
        files_a = sorted(glob.glob(os.path.join(src_a, '*.pt')))
        files_b = sorted(glob.glob(os.path.join(src_b, '*.pt')))

        # Build episode -> frames mapping for sensor_a
        episodes_a = {}
        for f in files_a:
            ep, fr = _parse_ep_fr(os.path.basename(f))
            if ep is None:
                continue
            episodes_a.setdefault(ep, []).append((fr, f))

        episodes_b = {}
        for f in files_b:
            ep, fr = _parse_ep_fr(os.path.basename(f))
            if ep is None:
                continue
            episodes_b.setdefault(ep, []).append((fr, f))

        # Sort frames within each episode
        for ep in episodes_a:
            episodes_a[ep].sort()
        for ep in episodes_b:
            episodes_b[ep].sort()

        orig_eps = sorted(episodes_a.keys())
        n_orig = len(orig_eps)
        print(f"  [{pair_name}/{obj_name}] {n_orig} episodes, "
              f"sensor_a={sensor_a}({len(files_a)} files), "
              f"sensor_b={sensor_b}({len(files_b)} files)")

        # Step 1: Copy originals
        for f in files_a:
            dst_path = os.path.join(dst_a, os.path.basename(f))
            shutil.copy2(f, dst_path)
        for f in files_b:
            dst_path = os.path.join(dst_b, os.path.basename(f))
            shutil.copy2(f, dst_path)

        # Step 2: Generate augmented copies
        for aug_idx in range(n_copies):
            for orig_ep in orig_eps:
                new_ep = n_orig + orig_ep * n_copies + aug_idx

                # Process sensor_a frames
                for fr, src_path in episodes_a.get(orig_ep, []):
                    rng = torch.Generator()
                    rng.manual_seed(seed + aug_idx * 10000 + orig_ep * 1000 + fr)
                    x, key = _load_raw(src_path)
                    x_aug = _add_noise(x, sensor_a, noise_scale, rng)
                    dst_path = os.path.join(dst_a, f'ep{new_ep:04d}_fr{fr:04d}.pt')
                    _save_raw(x_aug, key, dst_path)

                # Process sensor_b frames
                for fr, src_path in episodes_b.get(orig_ep, []):
                    rng = torch.Generator()
                    rng.manual_seed(seed + aug_idx * 10000 + orig_ep * 1000 + fr + 50000)
                    x, key = _load_raw(src_path)
                    x_aug = _add_noise(x, sensor_b, noise_scale, rng)
                    dst_path = os.path.join(dst_b, f'ep{new_ep:04d}_fr{fr:04d}.pt')
                    _save_raw(x_aug, key, dst_path)

        # Count output
        out_a = len(glob.glob(os.path.join(dst_a, '*.pt')))
        out_b = len(glob.glob(os.path.join(dst_b, '*.pt')))
        print(f"    -> {out_a} files ({sensor_a}), {out_b} files ({sensor_b})")


def main():
    parser = argparse.ArgumentParser(
        description='Augment paired dynamic data with Gaussian noise.')
    parser.add_argument('--src', required=True,
                        help='Source directory (e.g. latent_data_processed_dynamic/pnp)')
    parser.add_argument('--dst', required=True,
                        help='Destination directory (e.g. latent_data_processed_dynamic/pnp_aug)')
    parser.add_argument('--n_copies', type=int, default=9,
                        help='Number of noisy copies per episode (default: 9)')
    parser.add_argument('--noise_scale', type=float, default=0.05,
                        help='Noise std = noise_scale * sensor_data_std (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for reproducibility')
    args = parser.parse_args()

    if os.path.exists(args.dst):
        print(f"WARNING: {args.dst} already exists. Files may be overwritten.")

    print(f"Source:      {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Copies:      {args.n_copies} (total episodes = {5 * (1 + args.n_copies)})")
    print(f"Noise scale: {args.noise_scale} * sensor_data_std")
    print()

    for pair_name in _PAIRS:
        process_pair(pair_name, args.src, args.dst,
                     args.n_copies, args.noise_scale, args.seed)

    print("\nDone.")


if __name__ == '__main__':
    main()
