"""
Compute per-channel mean/std for every (object x split x modality) combination.

Splits:
    static  — paired data under latent_data_processed_object/{de,ef,fd}_pair/
              For each modality we pick one canonical pair to avoid double-counting:
                Daimon  → de_pair/daimon/<object>/
                eFlesh  → ef_pair/eflesh/<object>/
                FlexiTac→ fd_pair/flexi/<object>/
    dynamic — unpaired data under latent_data_processed_object/dynamic_data/<object>/<sensor>/

Objects:
    cylinder, plug (configurable via --objects).

Each .pt file is a dict like {'image': tensor}, {'magnetic': tensor}, or
{'flexi'/'flexitac': tensor}. Tensors are NOT yet z-scored.

Outputs YAML-ready stat blocks per (object, split):
    daimon_stats:   per-channel (length 3) mean/std
    eflesh_stats:   per-channel (length 15) mean/std
    flexitac_stats: scalar mean/std (averaged over all pixels)

Run from repo root:
    conda activate cmt
    python scripts/compute_per_object_stats.py
"""

import argparse
import glob
import os

import torch
from tqdm import tqdm
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, 'latent_data_processed_object')


def _extract(x):
    """Pull the raw tensor out of the saved dict format."""
    if isinstance(x, dict):
        if 'image' in x:
            return x['image']
        if 'magnetic' in x:
            return x['magnetic']
        if 'flexi' in x:
            return x['flexi']
        if 'flexitac' in x:
            return x['flexitac']
        raise ValueError(f"Unknown dict keys: {list(x.keys())}")
    return x


def _list_pt(folder):
    return sorted(glob.glob(os.path.join(folder, '*.pt')))


def _accumulate(files, per_channel_dim=None):
    """Stream files, return (mean, std).

    per_channel_dim:
        None  → scalar stats (mean/std over every element pooled)
        int k → keep dim k as channel axis, reduce all others
    """
    n = 0
    s = None     # sum
    s2 = None    # sum of squares
    desc = os.path.relpath(os.path.dirname(files[0]), DATA_ROOT) if files else ''
    for p in tqdm(files, desc=desc, leave=False):
        t = _extract(torch.load(p, map_location='cpu', weights_only=False)).float()
        if per_channel_dim is None:
            flat = t.reshape(-1)
            n += flat.numel()
            s = flat.sum() if s is None else s + flat.sum()
            s2 = (flat * flat).sum() if s2 is None else s2 + (flat * flat).sum()
        else:
            # move channel to dim 0, flatten the rest, accumulate per channel
            c = t.shape[per_channel_dim]
            perm = [per_channel_dim] + [i for i in range(t.dim()) if i != per_channel_dim]
            flat = t.permute(*perm).reshape(c, -1)
            n += flat.shape[1]
            cs = flat.sum(dim=1)
            cs2 = (flat * flat).sum(dim=1)
            s = cs if s is None else s + cs
            s2 = cs2 if s2 is None else s2 + cs2
    if n == 0:
        return None, None
    mean = s / n
    var = (s2 / n) - mean * mean
    std = var.clamp_min(0).sqrt()
    return mean, std


def _tolist(t):
    if t is None:
        return None
    if t.numel() == 1:
        return float(t.item())
    return [float(v) for v in t.tolist()]


def _dirs_for(obj, split):
    if split == 'static':
        return (
            os.path.join(DATA_ROOT, 'de_pair', 'daimon', obj),
            os.path.join(DATA_ROOT, 'ef_pair', 'eflesh', obj),
            os.path.join(DATA_ROOT, 'fd_pair', 'flexi', obj),
        )
    if split == 'dynamic':
        return (
            os.path.join(DATA_ROOT, 'dynamic_data', obj, 'daimon'),
            os.path.join(DATA_ROOT, 'dynamic_data', obj, 'eflesh'),
            os.path.join(DATA_ROOT, 'dynamic_data', obj, 'flexi'),
        )
    raise ValueError(split)


def _compute_block(d_files, e_files, f_files, tag):
    """Run the three sensors over the supplied file lists and return a stats block."""
    out = {}
    print(f"[{tag}] daimon:  {len(d_files):6d} files")
    m, s = _accumulate(d_files, per_channel_dim=0)
    if m is not None:
        out['daimon_stats'] = {'mean': _tolist(m), 'std': _tolist(s)}

    print(f"[{tag}] eflesh:  {len(e_files):6d} files")
    m, s = _accumulate(e_files, per_channel_dim=0)
    if m is not None:
        out['eflesh_stats'] = {'mean': _tolist(m), 'std': _tolist(s)}

    print(f"[{tag}] flexi:   {len(f_files):6d} files")
    m, s = _accumulate(f_files, per_channel_dim=None)
    if m is not None:
        out['flexitac_stats'] = {'mean': _tolist(m), 'std': _tolist(s)}
    return out


def compute_for(obj, split):
    """Per-object stats."""
    d_dir, e_dir, f_dir = _dirs_for(obj, split)
    return _compute_block(_list_pt(d_dir), _list_pt(e_dir), _list_pt(f_dir),
                          tag=f"{obj}/{split}")


def compute_combined(objects, split):
    """Pooled stats over all listed objects for a given split."""
    d_files, e_files, f_files = [], [], []
    for obj in objects:
        d_dir, e_dir, f_dir = _dirs_for(obj, split)
        d_files += _list_pt(d_dir)
        e_files += _list_pt(e_dir)
        f_files += _list_pt(f_dir)
    return _compute_block(d_files, e_files, f_files,
                          tag=f"combined({'+'.join(objects)})/{split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=['cylinder', 'plug'])
    parser.add_argument('--splits', nargs='+', default=['static', 'dynamic'])
    parser.add_argument('--out', default='per_object_stats.yaml',
                        help='Path (relative to repo root) for the YAML dump.')
    args = parser.parse_args()

    all_stats = {}
    for obj in args.objects:
        for split in args.splits:
            key = f"{obj}_{split}"
            print('=' * 60)
            print(f"=== {key} ===")
            print('=' * 60)
            all_stats[key] = compute_for(obj, split)

    # Pooled cross-object stats per split (used by joint-training configs).
    if len(args.objects) > 1:
        for split in args.splits:
            key = f"combined_{split}"
            print('=' * 60)
            print(f"=== {key} ({'+'.join(args.objects)}) ===")
            print('=' * 60)
            all_stats[key] = compute_combined(args.objects, split)

    out_path = os.path.join(REPO_ROOT, args.out)
    with open(out_path, 'w') as f:
        yaml.safe_dump(all_stats, f, sort_keys=False, default_flow_style=False)

    print('\n' + '=' * 60)
    print(f"Wrote → {out_path}")
    print('=' * 60)
    # Echo back to console too.
    print(yaml.safe_dump(all_stats, sort_keys=False, default_flow_style=False))


if __name__ == '__main__':
    main()
