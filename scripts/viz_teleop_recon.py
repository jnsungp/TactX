"""
Self + cross reconstruction visualization for teleop data.

For each teleop combo (two paired sensors X, Y), encodes both and shows:
    GT_X | self  dec_X(z_X) | cross dec_X(z_Y)
    GT_Y | self  dec_Y(z_Y) | cross dec_Y(z_X)

self  = decode a sensor's own latent  -> standard VAE recon.
cross = decode the OTHER sensor's latent through this sensor's decoder.
        This is the cross-modal test: it only works if the latent is shared
        (z_X and z_Y of the same contact land in the same place).

Uses the exact transform pipeline of plot_teleop_latents.py (transform table
+ rot180 + resize + z-score, contact-cropped).

Output: visualizations/teleop_recon/<combo>_recon.png  (one per combo,
6 rows = {X,Y} x {GT, self, cross}, columns = sample contact frames).

Usage:
    python scripts/viz_teleop_recon.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_static_run_scratch11_shear_efs3_20260517_043942 \
        --device cuda:0
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_teleop_latents import (
    apply_transform, contact_frames, preprocess_batch, _load_model,
    _COMBO_SPEC, _frame_files, _episode_dirs, _TELEOP_ROOT,
)


# ---------------------------------------------------------------------------
# Per-sensor single-sample renderers (GT and recon share these)
# ---------------------------------------------------------------------------

def _vis(ax, sensor, x):
    """Render one sample. x: tensor — Daimon (3,H,W) / eFlesh (15,) /
    FlexiTac (1,12,16) or (12,16)."""
    a = x.detach().cpu().float().numpy()
    if sensor == 'D':                                  # (3,H,W) -> RGB
        img = a.transpose(1, 2, 0)
        lo, hi = float(img.min()), float(img.max())
        if hi - lo > 1e-6:
            img = (img - lo) / (hi - lo)
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.axis('off')
    elif sensor == 'E':                                # (15,) -> bars
        v = a.reshape(-1)
        ax.bar(range(len(v)), v, color='steelblue')
        ax.tick_params(labelsize=5)
        ax.set_xticks([])
    else:                                              # (12,16) heatmap
        m = a.squeeze()
        ax.imshow(m, cmap='viridis', aspect='auto')
        ax.axis('off')


def _l1(a, b):
    return float((a - b).abs().mean().item())


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames(combo_dir, contact_threshold, all_frames, n, seed):
    """Pool contact-cropped frames across episodes, return n evenly-spaced."""
    pool = []
    for ep in _episode_dirs(combo_dir):
        fr = _frame_files(ep)
        if not all_frames:
            fr = contact_frames(fr, contact_threshold)
        pool += fr
    if not pool:
        return []
    if len(pool) <= n:
        return pool
    idx = np.linspace(0, len(pool) - 1, n).astype(int)
    return [pool[i] for i in idx]


@torch.no_grad()
def process_combo(combo, teleop_root, model, stats, image_size, device,
                  n, contact_threshold, all_frames, seed, out_dir):
    spec = _COMBO_SPEC[combo]
    sa, sb = spec['s0']['sensor'], spec['s1']['sensor']
    combo_dir = os.path.join(teleop_root, combo)
    if not os.path.isdir(combo_dir):
        print(f"[skip] missing {combo_dir}")
        return

    frames = _sample_frames(combo_dir, contact_threshold, all_frames, n, seed)
    if not frames:
        print(f"[skip] no frames for {combo}")
        return
    print(f"[{combo}] {sa}+{sb}: {len(frames)} sample frames")

    # transform -> model-ready tensors
    arr0, arr1 = [], []
    for fp in frames:
        with open(fp, 'rb') as fh:
            d = pickle.load(fh)
        arr0.append(apply_transform(d['s0'], spec['s0']))
        arr1.append(apply_transform(d['s1'], spec['s1']))
    x_a = preprocess_batch(arr0, sa, stats, image_size).to(device)
    x_b = preprocess_batch(arr1, sb, stats, image_size).to(device)

    # encode (mu == z in eval) and decode self / cross
    _, mu_a, _, _ = model._enc(x_a, sa)
    _, mu_b, _, _ = model._enc(x_b, sb)
    dec_a, dec_b = model._decoder_for(sa), model._decoder_for(sb)
    self_a, cross_a = dec_a(mu_a), dec_a(mu_b)        # dec_a(z_a), dec_a(z_b)
    self_b, cross_b = dec_b(mu_b), dec_b(mu_a)        # dec_b(z_b), dec_b(z_a)

    # per-recon mean L1 vs GT (recon targets the z-scored input)
    def _match(r, t):
        return r.squeeze(1) if (r.dim() == t.dim() + 1 and r.shape[1] == 1) else r
    err = {
        'self_a':  _l1(_match(self_a, x_a), x_a),
        'cross_a': _l1(_match(cross_a, x_a), x_a),
        'self_b':  _l1(_match(self_b, x_b), x_b),
        'cross_b': _l1(_match(cross_b, x_b), x_b),
    }

    rows = [
        (f'{sa} GT',                  sa, x_a),
        (f'{sa} self  L1={err["self_a"]:.3f}',  sa, self_a),
        (f'{sa} cross L1={err["cross_a"]:.3f}', sa, cross_a),
        (f'{sb} GT',                  sb, x_b),
        (f'{sb} self  L1={err["self_b"]:.3f}',  sb, self_b),
        (f'{sb} cross L1={err["cross_b"]:.3f}', sb, cross_b),
    ]
    m = len(frames)
    fig, axes = plt.subplots(6, m, figsize=(2.0 * m, 12), squeeze=False)
    for r, (label, sensor, batch) in enumerate(rows):
        for c in range(m):
            _vis(axes[r][c], sensor, batch[c])
            if c == 0:
                axes[r][c].set_ylabel(label, fontsize=8, rotation=0,
                                      ha='right', va='center', labelpad=38)
    fig.suptitle(f'{combo} — self / cross reconstruction (teleop, contact frames)',
                 fontsize=11)
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    out = os.path.join(out_dir, f'{combo}_recon.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}   "
          f"selfL1[{sa}]={err['self_a']:.3f} crossL1[{sa}]={err['cross_a']:.3f}  "
          f"selfL1[{sb}]={err['self_b']:.3f} crossL1[{sb}]={err['cross_b']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--weights', default='model.pt')
    parser.add_argument('--teleop_root', default=_TELEOP_ROOT)
    parser.add_argument('--stats_dataset', default='dataset_sachin_static')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n', type=int, default=8,
                        help='Sample contact frames per combo (columns).')
    parser.add_argument('--contact_threshold', type=float, default=0.12)
    parser.add_argument('--all_frames', action='store_true',
                        help='Sample from all frames, not just contact.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', default='visualizations/teleop_recon')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    teleop_root = (args.teleop_root if os.path.isabs(args.teleop_root)
                   else os.path.join(repo_root, args.teleop_root))
    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(repo_root, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(repo_root, 'config',
                           'datasets_config_sachin.yaml')) as f:
        dscfg = yaml.safe_load(f)[args.stats_dataset]
    stats = {'daimon': dscfg['daimon_stats'],
             'eflesh': dscfg['eflesh_stats'],
             'flexi': dscfg['flexitac_stats']}
    image_size = dscfg.get('image_size', 224)

    model = _load_model(args.checkpoint_dir, args.weights, args.device)
    print(f"Loaded {args.checkpoint_dir}/{args.weights}\n")

    for combo in _COMBO_SPEC:
        process_combo(combo, teleop_root, model, stats, image_size,
                      args.device, args.n, args.contact_threshold,
                      args.all_frames, args.seed, out_dir)

    print(f"\nRecon figures in {out_dir}/")


if __name__ == '__main__':
    main()
