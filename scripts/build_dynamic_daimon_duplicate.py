"""Build latent_data_processed_dynamic_daimon_duplicate.

Duplicates ch0 (normal deformation) of Daimon to ch1 and ch2 so the dynamic
data matches the deform-only-replicated Daimon format that dataset_concat
trains on. Non-daimon files (eflesh, flexi) are symlinked to save disk.

Layout copied:
    src = latent_data/latent_data_processed_dynamic/{pair}/{sensor}/{obj}/ep*_fr*.pt
    dst = latent_data/latent_data_processed_dynamic_daimon_duplicate/...

Daimon files (de_pair/daimon/, fd_pair/daimon/) are rewritten:
    new[0]=new[1]=new[2]=old[0]
All other files are symlinked.
"""
import argparse
import os
import shutil

import torch
from tqdm import tqdm


SRC_DEFAULT = 'latent_data/latent_data_processed_dynamic'
DST_DEFAULT = 'latent_data/latent_data_processed_dynamic_daimon_duplicate'

# (pair, sensor) tuples in the dataset.
ENTRIES = [
    ('de_pair', 'daimon'), ('de_pair', 'eflesh'),
    ('ef_pair', 'eflesh'), ('ef_pair', 'flexi'),
    ('fd_pair', 'flexi'),  ('fd_pair', 'daimon'),
]


def _rewrite_daimon(src_file, dst_file):
    x = torch.load(src_file, map_location='cpu', weights_only=False)
    if not torch.is_tensor(x):
        raise TypeError(f'{src_file}: expected tensor, got {type(x)}')
    if x.dim() != 3 or x.shape[0] != 3:
        raise ValueError(f'{src_file}: expected (3,H,W), got {tuple(x.shape)}')
    x = x[0:1].repeat(3, 1, 1).contiguous()
    torch.save(x, dst_file)


def _symlink(src_file, dst_file):
    if os.path.islink(dst_file) or os.path.exists(dst_file):
        os.remove(dst_file)
    os.symlink(os.path.abspath(src_file), dst_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC_DEFAULT)
    ap.add_argument('--dst', default=DST_DEFAULT)
    ap.add_argument('--objects', nargs='+', default=['plug', 'pnp', 'wipe'])
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f'Source not found: {args.src}')

    n_daimon = n_link = 0
    for pair, sensor in ENTRIES:
        is_daimon = (sensor == 'daimon')
        for obj in args.objects:
            src_obj = os.path.join(args.src, pair, sensor, obj)
            dst_obj = os.path.join(args.dst, pair, sensor, obj)
            if not os.path.isdir(src_obj):
                print(f'[skip] missing: {src_obj}')
                continue
            os.makedirs(dst_obj, exist_ok=True)
            files = sorted(f for f in os.listdir(src_obj) if f.endswith('.pt'))
            desc = f'{pair}/{sensor}/{obj} ({"rewrite" if is_daimon else "symlink"})'
            for fn in tqdm(files, desc=desc, leave=False):
                src_f = os.path.join(src_obj, fn)
                dst_f = os.path.join(dst_obj, fn)
                if is_daimon:
                    _rewrite_daimon(src_f, dst_f)
                    n_daimon += 1
                else:
                    _symlink(src_f, dst_f)
                    n_link += 1
    print(f'Done. daimon rewritten: {n_daimon}, symlinked: {n_link}')
    print(f'Output: {args.dst}')


if __name__ == '__main__':
    main()
