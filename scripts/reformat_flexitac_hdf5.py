#!/usr/bin/env python3
"""
Reformat existing FlexiTac HDF5 files from old 12x32 format to new 12x16 format.

Transforms applied (matching flexitac_sensor.py):
  Left  (tac0): np.fliplr → keep columns 0-15  → 12×16 = 192 flat
  Right (tac1): keep columns 16-31 → rotate 180° → 12×16 = 192 flat

Usage:
    conda activate crpo_teleop && cd ~/Workspace_cross_sensor

    python gello_software/gello/data_utils/reformat_flexitac_hdf5.py \
        --input_dir  /media/ExtHDD/sachin/teleop/flexi_flexi_object_pnp/processed \
        --output_dir /media/ExtHDD/sachin/teleop/flexi_flexi_object_pnp/processed_new_format
"""

import os
import argparse
import h5py
import numpy as np
from scipy.ndimage import rotate as ndimage_rotate
from pathlib import Path
from tqdm import tqdm


def transform_left(flat_384: np.ndarray) -> np.ndarray:
    """Apply left-port transform to a batch of flattened 12×32 arrays.

    Steps:
      1. Reshape (T, 384) → (T, 12, 32)
      2. Mirror horizontally (fliplr)
      3. Keep left half: columns 0-15 → (T, 12, 16)
      4. Flatten → (T, 192)
    """
    batch = flat_384.reshape(-1, 12, 32)
    mirrored = np.flip(batch, axis=2)  # fliplr along columns
    cropped = mirrored[:, :, :16]       # left half
    return cropped.reshape(-1, 192).astype(np.float32)


def transform_right(flat_384: np.ndarray) -> np.ndarray:
    """Apply right-port transform to a batch of flattened 12×32 arrays.

    Steps:
      1. Reshape (T, 384) → (T, 12, 32)
      2. Keep right half: columns 16-31 → (T, 12, 16)
      3. Rotate 180° (per frame)
      4. Clip to [0, 1], flatten → (T, 192)
    """
    batch = flat_384.reshape(-1, 12, 32)
    cropped = batch[:, :, 16:]  # right half: (T, 12, 16)

    rotated = np.zeros_like(cropped)
    for i in range(cropped.shape[0]):
        rotated[i] = ndimage_rotate(cropped[i], 180, reshape=False,
                                    order=1, mode='constant', cval=0.0)

    rotated = np.clip(rotated, 0.0, 1.0)
    return rotated.reshape(-1, 192).astype(np.float32)


def reformat_episode(input_path: Path, output_path: Path):
    """Read one HDF5 file, apply transforms, write a new one."""
    with h5py.File(input_path, 'r') as fin, h5py.File(output_path, 'w') as fout:
        tac0_old = fin['observation.tac0'][:]  # (T, 384)
        tac1_old = fin['observation.tac1'][:]  # (T, 384)

        tac0_new = transform_left(tac0_old)    # (T, 192)
        tac1_new = transform_right(tac1_old)   # (T, 192)

        # Copy all datasets, replacing tac0/tac1
        for name in fin.keys():
            if name == 'observation.tac0':
                fout.create_dataset(name, data=tac0_new, compression='gzip')
            elif name == 'observation.tac1':
                fout.create_dataset(name, data=tac1_new, compression='gzip')
            else:
                fin.copy(name, fout)

        # Copy and update attributes
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        # Update tactile metadata
        fout.attrs['tactile_dim'] = 192
        fout.attrs['tactile_data_shape'] = str((12, 16))
        fout.attrs['flexitac_reformat'] = 'left=fliplr+crop_left16, right=crop_right16+rot180'


def main():
    parser = argparse.ArgumentParser(
        description="Reformat FlexiTac HDF5 files from 12x32 to 12x16 (crop/mirror/rotate)"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with old-format episode_*.hdf5 files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for new-format episode_*.hdf5 files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(input_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        print(f"No episode_*.hdf5 files found in {input_dir}")
        return

    print(f"Reformatting {len(hdf5_files)} episodes: {input_dir} → {output_dir}")
    print(f"  Left transform:  fliplr → crop [:, :16] → 12×16 = 192")
    print(f"  Right transform: crop [:, 16:] → rotate 180° → 12×16 = 192")
    print()

    for hdf5_path in tqdm(hdf5_files, desc="Reformatting"):
        output_path = output_dir / hdf5_path.name
        reformat_episode(hdf5_path, output_path)

    print(f"\nDone! {len(hdf5_files)} episodes saved to {output_dir}")


if __name__ == "__main__":
    main()
