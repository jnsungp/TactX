"""
Sachin variant of plain paired-static tri-modal training — no dynamic-align.

This is the *true* "plain" baseline. The dynamic-align trainer
(train_dynamic_align_sachin.py) still runs the full dynamic forward
(`model.forward_self` on D/E/F) and the MMD computation every step even when
`gamma_mmd_max=0` — the zero-weighted term still builds its autograd graph and
holds dynamic-branch activations in memory until backward. That wastes RAM and
step time. This file calls `train_tri` instead, which never touches dynamic
data: paired-static tri-modal training only (recon + KL + InfoNCE over the
de / ef / fd pairs).

Paired dataset is loaded from config/datasets_config_sachin.yaml via
joint_embedding_learning.data.datasets_sachin, so per-pair geometric
corrections (rot180) are applied at __getitem__ time — identical to the
dynalign script's paired side.

Usage:
    python scripts/train_static_sachin.py \
        --device cuda:0 \
        --run_name plain_static \
        --objects cylinder plug

Latent-dimension sweep (overrides model_details['projection_dim']);
run_name gets a `_pd<N>` suffix automatically so sweep runs don't collide:
    python scripts/train_static_sachin.py --objects cylinder plug \
        --run_name pdsweep --projection_dim 32
"""

import argparse
import os
from datetime import datetime

import wandb
import yaml

from joint_embedding_learning.data.datasets_sachin import (
    get_paired_tri_datasets_sachin,
)
from joint_embedding_learning.training import train_tri


# Static paired objects. Joint cyl+plug by default (matches the Phase-3 setup).
DEFAULT_STATIC_OBJECTS = ['cylinder', 'plug']


def _load_dataset_details_sachin(dataset_name):
    """Read dataset_details from config/datasets_config_sachin.yaml."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, 'config', 'datasets_config_sachin.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if dataset_name not in cfg:
        raise KeyError(
            f"'{dataset_name}' not in {path}; available: {list(cfg.keys())}"
        )
    return cfg[dataset_name]


def main():
    parser = argparse.ArgumentParser(
        description='Train cross_modal_vae3 on paired-static data only '
                    '(Sachin per-pair rot180). No MMD, no cycle, no dynamic '
                    'data — the lean "plain" baseline.'
    )
    parser.add_argument('--dataset_name', type=str, default='dataset_sachin',
                        help='Paired tri-modal dataset name (from '
                             'datasets_config_sachin.yaml).')
    parser.add_argument('--model_name', type=str, default='cross_modal_vae3')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--run_name', type=str, default='plain_static')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--objects', nargs='+', default=DEFAULT_STATIC_OBJECTS,
                        help='Static paired objects (default: cylinder plug).')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Optional path to a checkpoint to warm-start from.')
    parser.add_argument('--projection_dim', type=int, default=None,
                        help='Override the shared latent dimension '
                             '(model_details["projection_dim"]). Used for '
                             'the latent-dim sweep. run_name gets a _pd<N> '
                             'suffix when this is set.')
    parser.add_argument('--infonce_warmup_start', type=int, default=None,
                        help='Epochs of recon-only training before the '
                             'alignment (InfoNCE) term turns on. Overrides '
                             'model_details. run_name gets a _ws<N> suffix.')
    parser.add_argument('--infonce_warmup_epochs', type=int, default=None,
                        help='Ramp width: epochs over which the alignment '
                             'weight alpha goes 0->1 after the start epoch. '
                             'Overrides model_details. run_name gets _we<N>.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total training epochs (overrides '
                             'model_details["epochs"], default 100).')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides '
                             'model_details["learning_rate"], default 1e-4). '
                             'Use a lower value (e.g. 1e-5) when finetuning '
                             'from --resume_from.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker processes (default 0 = '
                             'single-threaded, old behavior). Try 32 to '
                             'parallelize the per-item torch.load disk reads; '
                             'also enables pin_memory + persistent_workers.')
    parser.add_argument('--ef_stride', type=int, default=1,
                        help='Keep every Nth frame per episode in ef_pair. '
                             'ef (eFlesh+FlexiTac) is sampled ~2.7x faster '
                             'than the Daimon pairs; stride 3 ~equalizes the '
                             'three pairs. run_name gets a _efs<N> suffix.')
    parser.add_argument('--de_stride', type=int, default=1,
                        help='Per-episode frame stride for de_pair.')
    parser.add_argument('--fd_stride', type=int, default=1,
                        help='Per-episode frame stride for fd_pair.')
    # --- Debiased (false-negative-masked) InfoNCE -------------------------
    # Add --debiased at the end to switch the alignment term from plain
    # InfoNCE to the false-negative-masked variant: negatives come ONLY from
    # different (object, episode) frames, and zero-zero (no-contact/no-contact)
    # pairs are masked per sensor. Everything else (recon + KL + curriculum)
    # is identical, so the no-flag run is byte-for-byte the current baseline.
    parser.add_argument('--debiased', action='store_true',
                        help='Use false-negative-masked InfoNCE (different-'
                             'episode negatives + per-sensor zero-zero mask) '
                             'instead of plain InfoNCE. Requires '
                             '--contact_label_root.')
    parser.add_argument('--contact_label_root', type=str, default=None,
                        help='Root of per-episode contact labels from '
                             'scripts/build_contact_labels.py (e.g. '
                             'contact_labels/dataset_new). Used only with '
                             '--debiased.')
    parser.add_argument('--debiased_no_zero_mask', action='store_true',
                        help='Ablation: keep zero-zero pairs as negatives '
                             '(turn OFF the no-contact mask).')
    parser.add_argument('--debiased_no_episode_mask', action='store_true',
                        help='Ablation: allow same-(object,episode) frames as '
                             'negatives (turn OFF the different-episode mask).')
    # --- Zero-SupCon InfoNCE ---------------------------------------------
    # All no-contact frames in a batch become one positive group (SupCon-
    # style), in addition to the matched cross-sensor pair. No episode mask
    # and no zero-zero negative mask — just the extra positives. Same
    # contact-label dependency as --debiased, mutually exclusive with it.
    parser.add_argument('--zero_supcon', action='store_true',
                        help='Use zero-supcon InfoNCE: every no-contact frame '
                             'in the batch is positive to every other '
                             'no-contact frame (plus the matched cross-sensor '
                             'pair). Requires --contact_label_root.')
    # --- Intra-modal ±1 neighbor soft-positive InfoNCE --------------------
    # 2B-SupCon-out: matched cross-sensor pair (weight 1.0) + intra-modal
    # ±1 frame same-(object, episode) (weight `neighbor_weight`, default
    # 0.5). Negatives = every other off-diagonal entry. Auto-enables the
    # episode-aware batch sampler so ±1 frames co-occur in batches. No
    # contact labels needed. Mutually exclusive with --debiased / --zero_supcon.
    parser.add_argument('--neighbor', action='store_true',
                        help='Use intra-modal ±1 neighbor soft-positive '
                             'InfoNCE: matched pair (1.0) + same-(object, '
                             'episode) intra-modal ±1 (neighbor_weight). '
                             'Auto-enables episode-aware batch sampler.')
    parser.add_argument('--neighbor_weight', type=float, default=0.5,
                        help='Soft-positive weight for intra-modal |Δf|==1 '
                             '(default 0.5; matched pair stays at 1.0).')
    parser.add_argument('--neighbor_chunk_size', type=int, default=8,
                        help='EpisodeChunkBatchSampler chunk size for '
                             '--neighbor (default 8 contiguous frames per '
                             'episode chunk; reuses cwcl_chunk_size key).')
    # --- Temporal cosine alignment (auxiliary on top of baseline NT-Xent) -
    # Plain NT-Xent on matched cross-sensor pair (baseline-equal — negative
    # set / gradient unchanged) PLUS an auxiliary cosine alignment term on
    # intra-modal ±1 same-(object, episode) frames:
    #     L_tcos = (1 - cos(mu_a[t], mu_a[t±1])) + (1 - cos(mu_b[t], mu_b[t±1]))
    # weighted by `temporal_cos_weight` (default 0.05). Unlike --neighbor it
    # does NOT add ±1 frames as InfoNCE positives, so cross-modal contrast
    # gradient is not diluted. Mutually exclusive with --neighbor / --debiased
    # / --zero_supcon.
    parser.add_argument('--temporal_cos', action='store_true',
                        help='Baseline NT-Xent + intra-modal cosine alignment '
                             'on ±1 same-(object, episode) frames. Cross-modal '
                             'contrast untouched. Auto-enables episode-aware '
                             'batch sampler.')
    parser.add_argument('--temporal_cos_weight', type=float, default=0.05,
                        help='λ for the cosine alignment term (default 0.05). '
                             'NT-Xent stays at full strength regardless.')
    parser.add_argument('--temporal_cos_chunk_size', type=int, default=8,
                        help='EpisodeChunkBatchSampler chunk size for '
                             '--temporal_cos (default 8; reuses '
                             'cwcl_chunk_size key).')
    args = parser.parse_args()

    # Sweep suffixes (before timestamp) so sweep runs are distinct.
    if args.projection_dim is not None and args.run_name:
        args.run_name = f"{args.run_name}_pd{args.projection_dim}"
    if args.infonce_warmup_start is not None and args.run_name:
        args.run_name = f"{args.run_name}_ws{args.infonce_warmup_start}"
    if args.infonce_warmup_epochs is not None and args.run_name:
        args.run_name = f"{args.run_name}_we{args.infonce_warmup_epochs}"
    if args.ef_stride > 1 and args.run_name:
        args.run_name = f"{args.run_name}_efs{args.ef_stride}"
    if args.de_stride > 1 and args.run_name:
        args.run_name = f"{args.run_name}_des{args.de_stride}"
    if args.fd_stride > 1 and args.run_name:
        args.run_name = f"{args.run_name}_fds{args.fd_stride}"
    if args.debiased and args.run_name:
        tag = 'db'
        if args.debiased_no_zero_mask:
            tag += 'NOzero'
        if args.debiased_no_episode_mask:
            tag += 'NOep'
        args.run_name = f"{args.run_name}_{tag}"
    if args.zero_supcon and args.run_name:
        args.run_name = f"{args.run_name}_zsc"
    if args.neighbor and args.run_name:
        # _nbr<weight*100> e.g. nbr5 for 0.05, nbr10 for 0.10, nbr50 for 0.5.
        # Uses *100 so 0.05 and 0.1 don't collide as nbr1.
        args.run_name = f"{args.run_name}_nbr{int(round(args.neighbor_weight * 100))}"
    if args.temporal_cos and args.run_name:
        # _tcos<weight*100> e.g. tcos5 for 0.05, tcos10 for 0.10.
        args.run_name = f"{args.run_name}_tcos{int(round(args.temporal_cos_weight * 100))}"

    if sum([args.debiased, args.zero_supcon, args.neighbor, args.temporal_cos]) > 1:
        parser.error('--debiased / --zero_supcon / --neighbor / --temporal_cos '
                     'are mutually exclusive.')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp

    # Paired dataset details from sachin yaml (with per-pair extra_transforms).
    dataset_details = _load_dataset_details_sachin(args.dataset_name)

    # Model config from the upstream models_config.yaml.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, 'config', 'models_config.yaml')) as f:
        model_details = dict(yaml.safe_load(f)[args.model_name])

    # Latent-dim override for sweeps. projection_dim flows to the projection
    # heads (2*pd output) and all three decoders (pd in_dim) — single knob.
    if args.projection_dim is not None:
        model_details['projection_dim'] = args.projection_dim

    # Curriculum override: delay/ramp the alignment (InfoNCE) term so the
    # latent learns reconstruction first. See _infonce_alpha in training.py.
    if args.infonce_warmup_start is not None:
        model_details['infonce_warmup_start'] = args.infonce_warmup_start
    if args.infonce_warmup_epochs is not None:
        model_details['infonce_warmup_epochs'] = args.infonce_warmup_epochs
    if args.epochs is not None:
        model_details['epochs'] = args.epochs
    if args.lr is not None:
        model_details['learning_rate'] = args.lr
    model_details['num_workers'] = args.num_workers

    # Debiased InfoNCE: only touches model_details when --debiased is passed,
    # so the default run stays exactly the plain InfoNCE+recon baseline.
    if args.debiased:
        if not args.contact_label_root:
            parser.error('--debiased requires --contact_label_root '
                         '(e.g. contact_labels/dataset_new)')
        model_details['loss'] = 'debiased'
        model_details['contact_label_root'] = args.contact_label_root
        model_details['debiased_zero_mask'] = not args.debiased_no_zero_mask
        model_details['debiased_same_episode_mask'] = (
            not args.debiased_no_episode_mask)

    # Zero-SupCon InfoNCE: same contact-label requirement as --debiased, no
    # extra knobs. Stays off unless --zero_supcon is passed.
    if args.zero_supcon:
        if not args.contact_label_root:
            parser.error('--zero_supcon requires --contact_label_root '
                         '(e.g. contact_labels/dataset_new)')
        model_details['loss'] = 'zero_supcon'
        model_details['contact_label_root'] = args.contact_label_root

    # Neighbor-window soft-positive InfoNCE: 3-frame temporal window
    # (matched pair + ±1 neighbors). No contact labels needed; auto-enables
    # the episode-aware batch sampler via train_tri.
    if args.neighbor:
        model_details['loss'] = 'neighbor'
        model_details['neighbor_weight'] = args.neighbor_weight
        # train_tri reads `cwcl_chunk_size` to size EpisodeChunkBatchSampler.
        model_details['cwcl_chunk_size'] = args.neighbor_chunk_size

    # Temporal cosine alignment: baseline NT-Xent + auxiliary cos pull on
    # intra-modal ±1 same-(obj, ep) frames. Cross-modal contrast unchanged.
    if args.temporal_cos:
        model_details['loss'] = 'temporal_cos'
        model_details['temporal_cos_weight'] = args.temporal_cos_weight
        model_details['cwcl_chunk_size'] = args.temporal_cos_chunk_size

    project = 'CMTEJ_debug' if args.debug else 'CMTEJ_static_sachin'
    output_folder = (f"{args.model_name}_{dataset_details['dataset_type']}_"
                     f"sachin_{args.dataset_name}_{args.run_name}")
    wandb.init(project=project, name=output_folder, id=output_folder, resume=False)

    print(f"Run name: {args.run_name}")
    if args.debiased:
        print(f"Alignment loss: debiased InfoNCE  "
              f"(zero_zero_mask={model_details['debiased_zero_mask']}, "
              f"diff_episode_mask={model_details['debiased_same_episode_mask']})")
        print(f"Contact labels: {args.contact_label_root}")
    elif args.zero_supcon:
        print("Alignment loss: zero-supcon InfoNCE "
              "(all no-contact frames = one positive group)")
        print(f"Contact labels: {args.contact_label_root}")
    elif args.neighbor:
        print(f"Alignment loss: intra-modal ±1 neighbor soft-positive "
              f"InfoNCE (weight={args.neighbor_weight}, "
              f"chunk_size={args.neighbor_chunk_size})")
    elif args.temporal_cos:
        print(f"Alignment loss: baseline NT-Xent + temporal cos "
              f"alignment (λ={args.temporal_cos_weight}, "
              f"chunk_size={args.temporal_cos_chunk_size})")
    else:
        print("Alignment loss: infonce (baseline)")
    print(f"Paired dataset (sachin): {args.dataset_name}  objects={args.objects}")
    print(f"projection_dim (latent): {model_details['projection_dim']}")
    print(f"Checkpoint dir: checkpoints/{dataset_details['dataset_type']}/"
          f"{args.model_name}/{args.dataset_name}_run_{args.run_name}/")

    # Per-pair temporal subsampling (equalize sampling rate across pairs).
    pair_strides = {k: v for k, v in
                    (('de', args.de_stride), ('ef', args.ef_stride),
                     ('fd', args.fd_stride)) if v > 1}
    if pair_strides:
        print(f"Per-pair frame strides: {pair_strides}")

    # Paired (sachin) with per-pair geometric corrections.
    paired_train, paired_val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device=args.device, objects=args.objects,
        pair_strides=pair_strides or None,
    )

    train_tri(
        train_datasets=paired_train,
        validation_datasets={'val': paired_val},
        dataset_name=args.dataset_name,
        dataset_details=dataset_details,
        model_name=args.model_name,
        model_details=model_details,
        wandb=wandb,
        run_name=args.run_name,
        device=args.device,
        resume_from=args.resume_from,
    )


if __name__ == '__main__':
    main()
