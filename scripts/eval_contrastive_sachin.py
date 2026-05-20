"""
Evaluation for cross_modal_clr3 (tri-modal PLAIN-CONTRASTIVE) checkpoints
trained via scripts/train_contrastive_sachin.py.

This is the reconstruction-free counterpart of eval_cross_modal_vae3_sachin.py:
CrossModalCLR3 has no decoders and no (mu, logvar) head, so the recon
visualizations of the vae3 eval do not apply. Everything else — paired
cosine separation, cross-modal retrieval, and the t-SNE / MMD analyses —
is identical and reuses the helpers from eval_cross_modal_vae3_sachin.

CrossModalCLR3.forward_pair already exposes mu_a / mu_b aliases, so the
paired helpers work unchanged. For the dynamic (unpaired, per-modality)
section we attach a forward_self() that mirrors CrossModalVAE3's, built
from the model's own _enc(x, sensor) -> (h, z) projection.

What it produces (under --out_dir):
    cosine_dist_val.png         — pos vs neg histograms per pair (DE/EF/FD + DF transitive)
    tsne_paired_train.png       — t-SNE of paired-static train z (modality + pair links)
    tsne_paired_val.png         — same on val
    tsne_dynamic.png            — t-SNE of dynamic z alone (3 clusters should merge)
    tsne_static_vs_dynamic.png  — do dynamic latents land in the static manifold?
    mmd_static_vs_dynamic.txt   — numeric MMD per modality (lower = better domain match)
    retrieval.txt               — Top-1/5/10 cross-modal retrieval on paired val

Usage (matches the efs3 / static12 checkpoint):
    python scripts/eval_contrastive_sachin.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_clr3/dataset_sachin_static12_run_plain_contrastive_efs3_20260519_084825 \
        --ef_stride 3 \
        --device cuda:0
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.getcwd())
from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.data.datasets_sachin import get_paired_tri_datasets_sachin
from joint_embedding_learning.utils.models_setup import get_model

from scripts.eval_cross_modal_vae3_sachin import (
    _PAIRS,
    _SENSORS,
    cosine_pos_neg,
    extract_pair_mu,
    extract_self_mu,
    mmd_rbf_multi,
    plot_cosine_dist,
    plot_tsne_dynamic,
    plot_tsne_paired,
    plot_tsne_static_vs_dynamic,
    topk_retrieval,
    transitive_df_cosine,
)


# 12-object static set — matches DEFAULT_STATIC_OBJECTS in train_contrastive_sachin.py.
DEFAULT_STATIC_OBJECTS = [
    'circle', 'cylinder', 'ellipse', 'eraser', 'large_ball', 'line',
    'plug', 'small_ball', 'spam', 'triangle', 'shear', 'no_contact_threshold',
]


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a cross_modal_clr3 plain-contrastive checkpoint.'
    )
    parser.add_argument('--checkpoint_dir', required=True,
                        help='checkpoints/cross_modal_tri/cross_modal_clr3/'
                             'dataset_sachin_static12_run_<name>_<TS>')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset_name', default='dataset_sachin_static12')
    parser.add_argument('--dynamic_dataset_name', default='dataset_cross_modal_tri_dynamic')
    parser.add_argument('--objects', nargs='+', default=DEFAULT_STATIC_OBJECTS)
    parser.add_argument('--out_dir', default=None,
                        help='Default: <checkpoint_dir>/eval_sachin/')
    parser.add_argument('--max_tsne_points', type=int, default=2000)
    parser.add_argument('--max_mmd_points', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ef_stride', type=int, default=1,
                        help='Per-episode frame stride for ef_pair — set to '
                             'MATCH the training run (efs3 -> 3).')
    parser.add_argument('--de_stride', type=int, default=1)
    parser.add_argument('--fd_stride', type=int, default=1)
    parser.add_argument('--weights', default='model.pt')
    parser.add_argument('--skip_dynamic', action='store_true',
                        help='Paired-static eval only (no dynamic load, MMD, '
                             'or dynamic t-SNE).')
    args = parser.parse_args()

    if args.out_dir:
        out_dir = args.out_dir
    elif args.weights == 'model.pt':
        out_dir = os.path.join(args.checkpoint_dir, 'eval_sachin')
    else:
        stem = os.path.splitext(args.weights)[0]
        out_dir = os.path.join(args.checkpoint_dir, f'eval_{stem}')
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load model ──
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model('cross_modal_clr3', model_details).to(args.device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, args.weights),
                      map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded {args.checkpoint_dir}/{args.weights}  ->  out_dir={out_dir}")

    # CrossModalCLR3 has no forward_self; synthesize one from its own
    # _enc(x, sensor) -> (h, z) projection so extract_self_mu works unchanged.
    if not hasattr(model, 'forward_self'):
        def _forward_self(x, sensor, _m=model):
            _h, z = _m._enc(x, sensor)
            return {'mu': z}
        model.forward_self = _forward_self

    # ── Load paired (sachin) + dynamic datasets ──
    _pair_strides = {k: v for k, v in
                     (('de', args.de_stride), ('ef', args.ef_stride),
                      ('fd', args.fd_stride)) if v > 1}
    if _pair_strides:
        print(f"Per-pair frame strides: {_pair_strides}")
    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device='cpu', objects=args.objects,
        pair_strides=_pair_strides or None,
    )
    print('Paired val sizes:', {k: len(v) for k, v in paired_val.items()})
    if not args.skip_dynamic:
        dyn_train, _dv, _ = get_all_contrastive_datasets(
            args.dynamic_dataset_name, device='cpu',
        )
        print('Dynamic sizes:', {k: len(v) for k, v in dyn_train.items()})
    else:
        print('[skip_dynamic] dynamic-data section disabled')

    # ── Extract paired z (train + val) ──
    print('Extracting paired-static z...')
    mu_pairs_tr, mu_pairs_v = {}, {}
    for pk in _PAIRS:
        mu_pairs_tr[pk] = extract_pair_mu(paired_train[pk], pk, model, args.device)
        mu_pairs_v[pk] = extract_pair_mu(paired_val[pk], pk, model, args.device)

    # ── Extract dynamic z (per modality, unpaired) ──
    if not args.skip_dynamic:
        print('Extracting dynamic z...')
        dyn_mus = {s: extract_self_mu(dyn_train[s], s, model, args.device) for s in _SENSORS}

    static_mu_per_sensor = {
        'D': mu_pairs_tr['de'][0],
        'E': mu_pairs_tr['de'][1],
        'F': mu_pairs_tr['ef'][1],
    }

    # ── Cosine similarity per pair (val) ──
    print('Computing cosine distributions...')
    pos_de, neg_de = cosine_pos_neg(*mu_pairs_v['de'])
    pos_ef, neg_ef = cosine_pos_neg(*mu_pairs_v['ef'])
    pos_fd, neg_fd = cosine_pos_neg(*mu_pairs_v['fd'])
    pos_df, neg_df = transitive_df_cosine(
        mu_pairs_v['de'][0], mu_pairs_v['de'][1],
        mu_pairs_v['ef'][0], mu_pairs_v['ef'][1],
    )
    pairs_data = [
        ('DE direct', pos_de, neg_de, '#4C72B0'),
        ('EF direct', pos_ef, neg_ef, '#DD8452'),
        ('FD direct', pos_fd, neg_fd, '#55A868'),
        ('DF transitive', pos_df, neg_df, '#C44E52'),
    ]
    plot_cosine_dist(pairs_data, os.path.join(out_dir, 'cosine_dist_val.png'))

    # ── MMD static vs dynamic per modality ──
    if not args.skip_dynamic:
        print('Computing MMD static-vs-dynamic...')
        mmd_lines = ['MMD² static vs dynamic per modality (lower = better):']
        for s in _SENSORS:
            st = static_mu_per_sensor[s]
            dy = dyn_mus[s]
            n = min(args.max_mmd_points, len(st), len(dy))
            rng = np.random.RandomState(args.seed)
            ids_s = rng.choice(len(st), n, replace=False)
            ids_d = rng.choice(len(dy), n, replace=False)
            mmd = mmd_rbf_multi(st[ids_s], dy[ids_d])
            mmd_lines.append(f"  {s}: MMD² = {mmd:.6f}  "
                             f"(n_static={len(st)}, n_dynamic={len(dy)})")
        mmd_txt = '\n'.join(mmd_lines)
        print(mmd_txt)
        with open(os.path.join(out_dir, 'mmd_static_vs_dynamic.txt'), 'w') as f:
            f.write(mmd_txt + '\n')

    # ── Cross-modal retrieval on paired val ──
    print('Computing retrieval Top-k...')
    retr_lines = ['Cross-modal retrieval (paired val, Top-1/5/10):']
    for pk in _PAIRS:
        ma, mb = mu_pairs_v[pk]
        ab = topk_retrieval(ma, mb)
        ba = topk_retrieval(mb, ma)
        retr_lines.append(f"  {pk} a→b: " + '  '.join(f"top{k}={v:.4f}" for k, v in ab.items()))
        retr_lines.append(f"  {pk} b→a: " + '  '.join(f"top{k}={v:.4f}" for k, v in ba.items()))
    retr_txt = '\n'.join(retr_lines)
    print(retr_txt)
    with open(os.path.join(out_dir, 'retrieval.txt'), 'w') as f:
        f.write(retr_txt + '\n')

    # ── t-SNE plots ──
    print('Computing t-SNE plots...')
    plot_tsne_paired(mu_pairs_tr, 'train', os.path.join(out_dir, 'tsne_paired_train.png'),
                     n_max=args.max_tsne_points)
    plot_tsne_paired(mu_pairs_v, 'val', os.path.join(out_dir, 'tsne_paired_val.png'),
                     n_max=args.max_tsne_points)
    if not args.skip_dynamic:
        plot_tsne_dynamic(dyn_mus, os.path.join(out_dir, 'tsne_dynamic.png'),
                          n_max=args.max_tsne_points)
        plot_tsne_static_vs_dynamic(
            static_mu_per_sensor, dyn_mus,
            os.path.join(out_dir, 'tsne_static_vs_dynamic.png'),
            n_max=min(args.max_tsne_points, 1500),
        )

    print(f"\nDone. All outputs under: {out_dir}/")


if __name__ == '__main__':
    main()
