"""
Cos sim on CONTACT frames in the dynamic paired dataset.

For each pair (de, ef, fd), select frames where BOTH sensors are in contact
and compute the cosine similarity between mu_a[i] and mu_b[i] (same-frame
positive pair). Reports mean ± std per pair and overall.

Two ways to define "contact":
  --mode binary  (default)  →  contact_a == 1 AND contact_b == 1
  --mode score              →  norm_a >= thr AND norm_b >= thr  (set --score_threshold)

The dataset is built directly from --data_root; no new yaml entry needed.
Sensor stats / transforms / pair layout are inherited from the yaml entry
the model was TRAINED on (--stats_from, default dataset_new) and only
data_root + val_episode_ids are overridden.

Expected on-disk layout (matches the static datasets):
    <data_root>/de_pair/{daimon,eflesh}/<object>/ep****_fr****.pt
    <data_root>/ef_pair/{eflesh,flexi}/<object>/ep****_fr****.pt
    <data_root>/fd_pair/{flexi,daimon}/<object>/ep****_fr****.pt

Contact labels at:
    <contact_label_root>/{de_pair,ef_pair,fd_pair}/<object>/ep****.pt
    (keys: contact_a, contact_b, norm_a, norm_b, frames, episode, …)

Usage:
    python scripts/eval_dynamic_contact_cossim.py \
  --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_concat_run_final_zsc_20260526_030030 \
  --data_root latent_data/latent_data_processed_dynamic \
  --objects plug wipe pnp \
  --contact_label_root contact_labels/dynamic \
  --stats_from dataset_concat \
  --score_threshold 0.5 \
  --device cuda:0


    # frames where both normalized scores are >= 0.3 instead of binary contact:
    python scripts/eval_dynamic_contact_cossim.py ... \
        --mode score --score_threshold 0.3
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from joint_embedding_learning.data.datasets_sachin import (
    get_cross_modal_tri_datasets_sachin,
)
from joint_embedding_learning.utils.models_setup import get_model

_PAIRS = ('de', 'ef', 'fd')
_PAIR_SENSORS = {'de': ('D', 'E'), 'ef': ('E', 'F'), 'fd': ('F', 'D')}
_PAIR_DIR = {'de': 'de_pair', 'ef': 'ef_pair', 'fd': 'fd_pair'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(d):
    while isinstance(d, Subset):
        d = d.dataset
    return d


@torch.no_grad()
def extract_pair_mu(dataset, pair_name, model, device, batch_size=128):
    """Return (mu_a, mu_b, obj_ids, ep_ids, fr_ids)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    mu_a_all, mu_b_all, obj_all, ep_all, fr_all = [], [], [], [], []
    for (xa, xb), labels in tqdm(loader, desc=f"paired_{pair_name}", leave=False):
        out = model.forward_pair(xa.to(device), xb.to(device), pair_name)
        mu_a_all.append(out['mu_a'].cpu())
        mu_b_all.append(out['mu_b'].cpu())
        obj_id, ep_id, fr_id = labels
        obj_all.append(obj_id.cpu())
        ep_all.append(ep_id.cpu())
        fr_all.append(fr_id.cpu())
    return (torch.cat(mu_a_all), torch.cat(mu_b_all),
            torch.cat(obj_all), torch.cat(ep_all), torch.cat(fr_all))


def _load_pair_labels(label_root, pair_name, object_names):
    """{(obj_id, ep, fr): (contact_a, contact_b, norm_a, norm_b)}.

    obj_id indexes into `object_names` — pass the dataset's index→name list so
    keys line up with meta tensors from the dataloader."""
    pair_dir = _PAIR_DIR[pair_name]
    out = {}
    for obj_id, obj in enumerate(object_names):
        ep_dir = os.path.join(label_root, pair_dir, obj)
        if not os.path.isdir(ep_dir):
            continue
        for f in sorted(glob.glob(os.path.join(ep_dir, 'ep*.pt'))):
            d = torch.load(f, map_location='cpu', weights_only=False)
            ep = int(d['episode'])
            frames = np.asarray(d['frames'])
            ca = np.asarray(d['contact_a']).astype(np.int64)
            cb = np.asarray(d['contact_b']).astype(np.int64)
            na = np.asarray(d['norm_a']).astype(np.float32)
            nb = np.asarray(d['norm_b']).astype(np.float32)
            for i, fr in enumerate(frames):
                out[(obj_id, ep, int(fr))] = (
                    int(ca[i]), int(cb[i]), float(na[i]), float(nb[i]))
    return out


def contact_mask(meta, labels, mode, threshold):
    """Boolean keep mask + count of frames missing a label."""
    obj_t, ep_t, fr_t = meta
    obj_np, ep_np, fr_np = obj_t.numpy(), ep_t.numpy(), fr_t.numpy()
    n = obj_t.shape[0]
    keep = np.zeros(n, dtype=bool)
    missing = 0
    for i in range(n):
        key = (int(obj_np[i]), int(ep_np[i]), int(fr_np[i]))
        if key not in labels:
            missing += 1
            continue
        ca, cb, na, nb = labels[key]
        if mode == 'binary':
            keep[i] = (ca == 1 and cb == 1)
        else:  # 'score'
            keep[i] = (na >= threshold and nb >= threshold)
    return keep, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Contact-frame cos sim on a paired dynamic dataset')
    ap.add_argument('--checkpoint_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--data_root', default='latent_data/latent_data_processed_dynamic',
                    help='Paired dataset root (must contain de_pair/ ef_pair/ fd_pair/).')
    ap.add_argument('--objects', nargs='+', default=['plug', 'pnp', 'wipe'])
    ap.add_argument('--contact_label_root', default='contact_labels/dynamic')
    ap.add_argument('--stats_from', default='dataset_new',
                    help='Yaml entry to inherit sensor stats / transforms / pairs '
                         'from. Use the SAME entry the model was trained on so the '
                         'encoder sees in-distribution normalization.')
    ap.add_argument('--val_episode_ids', type=int, nargs='+', default=[4],
                    help='Episode ids to call "val" (rest = train). Default [4] '
                         'matches dataset_sachin_dynamic_paired.')
    ap.add_argument('--split', default='all', choices=['train', 'val', 'all'],
                    help="Split to evaluate on. Default 'all' pools train+val "
                         "(dynamic data is small).")
    ap.add_argument('--mode', default='binary', choices=['binary', 'score'])
    ap.add_argument('--score_threshold', type=float, default=None,
                    help='Passing this implies --mode score. Default (when '
                         'score mode is used) is 0.5.')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    # Convenience: passing --score_threshold alone implies score mode (otherwise
    # the threshold is silently ignored under the default --mode binary).
    if args.score_threshold is not None and args.mode == 'binary':
        print('[note] --score_threshold given → switching --mode binary → score')
        args.mode = 'score'
    if args.score_threshold is None:
        args.score_threshold = 0.5

    out_dir = args.out_dir or os.path.join(
        args.checkpoint_dir, 'eval_dynamic_contact_cossim')
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── model ──
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model('cross_modal_vae3', model_details).to(args.device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'model.pt'),
                      map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded {args.checkpoint_dir}/model.pt")

    # ── dataset details: inherit stats from --stats_from, override data_root +
    #    val_episode_ids so the split is sensible for plug/pnp/wipe ──
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo_root, 'config', 'datasets_config_sachin.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if args.stats_from not in cfg:
        raise SystemExit(
            f"'{args.stats_from}' not in {cfg_path}; "
            f"available: {list(cfg.keys())}")
    details = dict(cfg[args.stats_from])
    details['data_root'] = args.data_root
    details['episode_val_split'] = True
    details['val_episode_ids'] = list(args.val_episode_ids)
    train, val, _ = get_cross_modal_tri_datasets_sachin(
        args.stats_from, details, device='cpu', objects=args.objects)
    print('train sizes:', {k: len(v) for k, v in train.items()})
    print('val sizes:  ', {k: len(v) for k, v in val.items()})

    # ── pick split ──
    if args.split == 'train':
        datasets = train
    elif args.split == 'val':
        datasets = val
    else:
        datasets = {pk: ConcatDataset([train[pk], val[pk]]) for pk in _PAIRS}

    # ── object_names from the base dataset (consistent across pairs) ──
    object_names_per_pair = {
        pk: list(_unwrap(train[pk]).object_names) for pk in _PAIRS}

    # ── run ──
    crit_str = ('contact_a==1 AND contact_b==1' if args.mode == 'binary'
                else f'norm_a >= {args.score_threshold} AND norm_b >= {args.score_threshold}')
    header = [
        'Cos sim on CONTACT frames (dynamic dataset, paired positives):',
        f'  data_root:  {args.data_root}',
        f'  objects:    {args.objects}',
        f'  stats_from: {args.stats_from}',
        f'  split:      {args.split}',
        f'  criterion:  [{args.mode}] {crit_str}',
        f'  labels:     {args.contact_label_root}',
        '',
    ]
    print('\n'.join(header))

    lines, all_cos = [], []
    for pk in _PAIRS:
        mu_a, mu_b, oa, ea, fa = extract_pair_mu(
            datasets[pk], pk, model, args.device)
        labels = _load_pair_labels(
            args.contact_label_root, pk, object_names_per_pair[pk])
        if not labels:
            lines.append(f"  {pk}: NO LABELS under {args.contact_label_root}/{_PAIR_DIR[pk]} — skipped")
            continue
        keep, missing = contact_mask(
            (oa, ea, fa), labels, args.mode, args.score_threshold)
        n_keep, n_tot = int(keep.sum()), len(keep)
        if n_keep == 0:
            lines.append(
                f"  {pk}: 0 contact frames (of {n_tot}, "
                f"missing_label={missing}) — skipped")
            continue
        kt = torch.from_numpy(keep)
        a_n = F.normalize(mu_a[kt], dim=1)
        b_n = F.normalize(mu_b[kt], dim=1)
        cos = (a_n * b_n).sum(dim=1).cpu().numpy()
        all_cos.append(cos)
        lines.append(
            f"  {pk}: mean cos = {cos.mean():+.4f}  std = {cos.std():.4f}  "
            f"(n_contact={n_keep}/{n_tot}, missing_label={missing})")

    if all_cos:
        merged = np.concatenate(all_cos, axis=0)
        lines.append(
            f"  overall: mean cos = {merged.mean():+.4f}  "
            f"std = {merged.std():.4f}  (n={len(merged)})")
    else:
        lines.append("  overall: no contact frames found in any pair.")

    body = '\n'.join(lines)
    print(body)

    suffix = ('binary' if args.mode == 'binary'
              else f'score_thr{args.score_threshold:g}')
    out_path = os.path.join(
        out_dir, f'dynamic_contact_cossim_{args.split}_{suffix}.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(header) + '\n' + body + '\n')
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
