"""
Cosine similarity on zero-contact val pairs.

For each pair (de, ef, fd) in the val split, select frames where BOTH
sensor contact scores are <= score_threshold (default 0.0 = exactly zero),
then compute cosine similarity between mu_a and mu_b across those frames
and report the mean (per pair + overall).

Uses the contact-score labels (norm_a/norm_b or scores_a/scores_b), NOT
the binary contact_pair label. Reuses the dataset/extraction helpers from
scripts/eval_cross_modal_vae3_new.py to keep the val split identical.

Usage:
    python scripts/eval_zero_contact_cossim.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/0_TACTX_concat \
        --dataset_name dataset_new_concat \
        --objects circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle \
        --contact_label_root contact_labels/dataset_new_concat \
        --device cuda:0

    # relax the zero-contact threshold:
    python scripts/eval_zero_contact_cossim.py ... --score_threshold 0.05
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.getcwd())
from joint_embedding_learning.data.datasets_sachin import (
    get_paired_tri_datasets_sachin,
)
from joint_embedding_learning.utils.models_setup import get_model


def _load_eval_module():
    """Load helpers from scripts/eval_cross_modal_vae3_new.py without
    requiring scripts/ to be a package."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'eval_cross_modal_vae3_new.py')
    spec = importlib.util.spec_from_file_location('eval_cross_modal_vae3_new', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ev = _load_eval_module()
_PAIRS = _ev._PAIRS
_PAIR_SENSORS = _ev._PAIR_SENSORS
extract_pair_mu = _ev.extract_pair_mu
_unwrap_dataset = _ev._unwrap_dataset
_load_pair_score_labels = _ev._load_pair_score_labels


def zero_score_mask(meta, label_map, threshold=0.0):
    """Boolean mask over samples: True where (score_a <= thr) AND (score_b <= thr).

    `meta` is (obj_id_tensor, ep_id_tensor, fr_id_tensor) as returned by
    extract_pair_mu(return_meta=True). Returns (mask, n_missing_label)."""
    obj_t, ep_t, fr_t = meta
    n = obj_t.shape[0]
    keep = np.zeros(n, dtype=bool)
    obj_np = obj_t.numpy()
    ep_np = ep_t.numpy()
    fr_np = fr_t.numpy()
    missing = 0
    for i in range(n):
        key = (int(obj_np[i]), int(ep_np[i]), int(fr_np[i]))
        if key not in label_map:
            missing += 1
            continue
        sa, sb = label_map[key]
        if sa <= threshold and sb <= threshold:
            keep[i] = True
    return keep, missing


def main():
    parser = argparse.ArgumentParser(
        description='Cosine sim on zero-contact val pairs.'
    )
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset_name', default='dataset_sachin_new')
    parser.add_argument('--objects', nargs='+', default=['small_ball'])
    parser.add_argument('--ef_stride', type=int, default=1)
    parser.add_argument('--de_stride', type=int, default=1)
    parser.add_argument('--fd_stride', type=int, default=1)
    parser.add_argument('--contact_label_root', required=True,
                        help='Root of per-episode contact label .pt files '
                             '(layout: <root>/<pair_dir>/<obj>/ep****.pt).')
    parser.add_argument('--score_key', default='norm',
                        choices=['norm', 'scores'],
                        help='norm = per-episode min-max normalized score in [0,1]; '
                             'scores = raw mean-abs-diff score.')
    parser.add_argument('--score_threshold', type=float, default=1.0,
                        help='Frame is zero-contact if BOTH sensor scores '
                             '<= this value. Default 0.0 is strict (with '
                             'score_key=norm picks the per-episode min frames).')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--contrastive_only', action='store_true',
                        help='Load a CrossModalCLR3 checkpoint instead of VAE3.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'eval_new')
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = 'cross_modal_clr3' if args.contrastive_only else 'cross_modal_vae3'
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model(model_name, model_details).to(args.device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'model.pt'),
                      map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded {args.checkpoint_dir}/model.pt  (model={model_name})")

    pair_strides = {k: v for k, v in
                    (('de', args.de_stride), ('ef', args.ef_stride),
                     ('fd', args.fd_stride)) if v > 1}
    _, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device='cpu', objects=args.objects,
        pair_strides=pair_strides or None,
    )
    print('Paired val sizes:', {k: len(v) for k, v in paired_val.items()})

    object_names_per_pair = {
        pk: list(_unwrap_dataset(paired_val[pk]).object_names) for pk in _PAIRS
    }

    print(f"\n=== Cosine sim on zero-contact val pairs "
          f"(score_key={args.score_key}, threshold={args.score_threshold}) ===")
    out_lines = [
        f'Cosine sim between mu_a and mu_b on val pairs where BOTH '
        f'{args.score_key} scores <= {args.score_threshold}:',
        f'  label_root: {args.contact_label_root}',
        f'  objects: {args.objects}',
    ]
    all_cos = []

    for pk in _PAIRS:
        mu_a, mu_b, oa, ea, fa = extract_pair_mu(
            paired_val[pk], pk, model, args.device, return_meta=True,
        )
        labels = _load_pair_score_labels(
            args.contact_label_root, pk, object_names_per_pair[pk],
            score_key=args.score_key,
        )
        if not labels:
            out_lines.append(f"  {pk}: no labels under {args.contact_label_root} — skipped")
            continue

        keep, missing = zero_score_mask(
            (oa, ea, fa), labels, threshold=args.score_threshold,
        )
        n_keep = int(keep.sum())
        n_total = len(keep)
        if n_keep == 0:
            out_lines.append(
                f"  {pk}: 0 zero-contact frames (of {n_total}, "
                f"missing_label={missing}) — skipped"
            )
            continue

        ka = torch.from_numpy(keep)
        a_n = F.normalize(mu_a[ka], dim=1)
        b_n = F.normalize(mu_b[ka], dim=1)
        cos = (a_n * b_n).sum(dim=1).cpu().numpy()
        all_cos.append(cos)
        out_lines.append(
            f"  {pk}: mean cos = {cos.mean():+.4f}  std = {cos.std():.4f}  "
            f"(n_zero={n_keep}/{n_total}, missing_label={missing})"
        )

    if all_cos:
        merged = np.concatenate(all_cos, axis=0)
        out_lines.append(
            f"  overall (all zero-contact val pairs): mean cos = "
            f"{merged.mean():+.4f}  std = {merged.std():.4f}  (n={len(merged)})"
        )
    else:
        out_lines.append('  overall: no zero-contact frames found in any pair.')

    out_txt = '\n'.join(out_lines)
    print(out_txt)
    metrics_path = os.path.join(out_dir, 'zero_contact_cossim.txt')
    with open(metrics_path, 'w') as f:
        f.write(out_txt + '\n')
    print(f"\nSaved: {metrics_path}")


if __name__ == '__main__':
    main()
