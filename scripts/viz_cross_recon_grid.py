"""
3x4 cross-recon grid blocks for a cross_modal_vae3 checkpoint (e.g. 0_TACTX),
laid out as several blocks side by side — one block per object.

Each block is a 3x4 grid for ONE object, all from the SAME physical contact
(same object / episode / frame, so the three sensors are genuinely paired):

        |   GT    |   Self    |   Cross    |   Cross    |
    D   |  GT_D   |  D(z_D)   |  D(z_E)    |  D(z_F)    |
    E   |  GT_E   |  E(z_E)   |  E(z_D)    |  E(z_F)    |
    F   |  GT_F   |  F(z_F)   |  F(z_D)    |  F(z_E)    |

Rows are the three sensors  D=Daimon  E=eFlesh  F=FlexiTac.
Column 1 is ground truth, column 2 is self-reconstruction (decode a sensor's own
latent), columns 3-4 are cross-reconstructions (decode each OTHER sensor's latent
through this sensor's decoder). Every cell title reads "<decoder>←<latent source>".

Triplets are formed by intersecting the de and ef VAL splits on (object, episode,
frame): D & E come from the de pair, F from the ef pair, so all three modalities
in a block correspond to the same grasp. The val split holds out whole episodes,
so these contacts were never seen during training.

Usage:
    python scripts/viz_cross_recon_grid.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/0_TACTX \
        --objects small_ball cylinder circle
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import Subset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from joint_embedding_learning.utils.models_setup import get_model
from joint_embedding_learning.data.datasets_sachin import get_paired_tri_datasets_sachin

# Sensor order used for the rows and the cross columns.
ORDER = ('D', 'E', 'F')
SENSOR_NAME = {'D': 'Daimon', 'E': 'eFlesh', 'F': 'FlexiTac'}


# ---------------------------------------------------------------------------
# Triplet assembly across the paired val splits
# ---------------------------------------------------------------------------

def _subset_keys(subset):
    """{(obj_name, episode, frame): local_idx} for one (possibly Subset) val ds."""
    if isinstance(subset, Subset):
        base, idxs = subset.dataset, list(subset.indices)
    else:
        base, idxs = subset, list(range(len(subset)))
    out = {}
    for li, gi in enumerate(idxs):
        key = (base.object_names[base.object_per_sample[gi]],
               int(base.episode_per_sample[gi]),
               int(base.frame_per_sample[gi]))
        out[key] = li
    return out


def build_triplet_bank(val):
    """Return {key: {'D': x, 'E': x, 'F': x, 'idx': (de_i, ef_i)}}.

    D & E are pulled from the de pair (sensor_a, sensor_b); F from the ef pair
    (sensor_b). A key is a valid triplet only if present in BOTH de and ef val
    splits — guaranteeing the three modalities share one (object, ep, frame).
    Inputs are loaded lazily (only chosen keys), so we store indices here.
    """
    de_keys = _subset_keys(val['de'])
    ef_keys = _subset_keys(val['ef'])
    common = set(de_keys) & set(ef_keys)
    return {k: (de_keys[k], ef_keys[k]) for k in common}


def _flexi_for(ds, mode, idx):
    """The Flexi pressure grid for a candidate frame (used for contact scoring).
    mode='fd' -> fd pair sensor_a; otherwise -> ef pair sensor_b."""
    if mode == 'fd':
        (x_f, _), _ = ds['fd'][idx]
    else:
        (_, x_f), _ = ds['ef'][idx]
    return x_f


def contact_score(x_f):
    """Contact intensity from the Flexi grid: ~uniform at rest, a localized
    high-pressure bump on contact, so its spatial spread (std) peaks at maximum
    contact within an episode."""
    return float(x_f.detach().cpu().float().std())


def load_block(ds, model, device, mode, idx):
    """Return (gt, z): per-sensor GT tensors and latents for one contact.

    mode='tri'  D,E come from the de pair, F from the ef pair; latents are
                encoded from the three real inputs.
    mode='fd'   D,F are REAL from the fd pair (same contact, fd orientation —
                note Flexi is NOT rot180'd here, unlike the ef pair). The fd
                pair has no eFlesh, so eFlesh is GENERATED: z_E := z_D and the
                eFlesh "GT" cell shows E(z_D).
    """
    if mode == 'tri':
        de_i, ef_i = idx
        (x_d, x_e), _ = ds['de'][de_i]   # de = (daimon, eflesh)
        (_, x_f), _ = ds['ef'][ef_i]     # ef = (eflesh, flexi)
        gt = {'D': x_d, 'E': x_e, 'F': x_f}
        z = {s: model._enc(gt[s].unsqueeze(0).to(device), s)[3] for s in ORDER}
    else:  # 'fd'
        (x_f, x_d), _ = ds['fd'][idx]    # fd = (flexi, daimon)
        z_d = model._enc(x_d.unsqueeze(0).to(device), 'D')[3]
        z_f = model._enc(x_f.unsqueeze(0).to(device), 'F')[3]
        z = {'D': z_d, 'E': z_d, 'F': z_f}                  # z_E := z_D
        x_e = model._decoder_for('E')(z['E']).squeeze(0).detach().cpu()
        gt = {'D': x_d, 'E': x_e, 'F': x_f}                 # eFlesh GT = E(z_D)
    return gt, z


# ---------------------------------------------------------------------------
# Per-sensor display (GT range is reused for the recons so colors are comparable)
# ---------------------------------------------------------------------------

def disp(ax, sensor, x, vrange):
    x = x.detach().cpu().float()
    if sensor == 'D':
        if x.dim() == 4:
            x = x[0]
        vmin, vmax = vrange
        img = ((x - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)
        if img.shape[0] == 3:
            ax.imshow(img.permute(1, 2, 0).numpy())
        else:
            ax.imshow(img.squeeze(0).numpy(), cmap='gray')
    elif sensor == 'E':
        v = x.reshape(-1)[:15].numpy()               # 5 magnets x 3 axes = 15
        a = vrange
        ax.bar(range(len(v)), v, color='steelblue', width=0.85)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlim(-0.6, len(v) - 0.4)
        ax.set_ylim(-a, a)
    else:  # 'F' — square pixels (aspect='equal'), not stretched
        while x.dim() > 2:
            x = x.squeeze(0)
        vmin, vmax = vrange
        ax.imshow(x.numpy(), cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])


def value_range(sensor, imgs):
    """Shared display range over all cells in a row (GT + recons) so colors /
    bar heights are directly comparable and never clip."""
    vals = torch.cat([t.detach().cpu().float().reshape(-1) for t in imgs])
    if sensor == 'E':
        return float(vals.abs().max()) + 1e-8        # symmetric bar y-limits
    return (float(vals.min()), float(vals.max()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint_dir', required=True,
                    help='e.g. checkpoints/cross_modal_tri/cross_modal_vae3/0_TACTX')
    ap.add_argument('--checkpoint_file', default='model.pt')
    ap.add_argument('--dataset_name', default='dataset_new',
                    help='Entry in config/datasets_config_sachin.yaml (0_TACTX -> dataset_new)')
    ap.add_argument('--objects', nargs='+', default=['small_ball', 'cylinder', 'circle'],
                    help='One object per block, left to right.')
    ap.add_argument('--split', choices=['train', 'val'], default='val',
                    help='Which episode split to pull the triplets from.')
    ap.add_argument('--gen_eflesh', nargs='*', default=[],
                    help='Objects for which to take GT Daimon+Flexi from the fd '
                         'pair and GENERATE eFlesh (z_E := z_D). Others use the '
                         'real de∩ef triplet. e.g. --gen_eflesh plane circle')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0,
                    help='Default per-object seed (used when --seeds is omitted).')
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='Per-object seeds, one per --objects entry, so you can '
                         'lock a good object and re-roll only the others. A '
                         'single value applies to all. Each object is seeded '
                         'independently. Defaults to --seed.')
    ap.add_argument('--out', default=None,
                    help='default: <checkpoint_dir>/recon_val/cross_recon_grid.png')
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.checkpoint_dir, 'recon_val',
                                        'cross_recon_grid.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── model ──
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model('cross_modal_vae3', model_details).to(args.device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_file),
                      map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()                              # eval => z = mu (no sampling noise)
    print(f"Loaded {args.checkpoint_dir}/{args.checkpoint_file}")

    # ── paired splits (same train/val episode logic as training) ──
    train, val, _ = get_paired_tri_datasets_sachin(
        args.dataset_name, device='cpu', objects=args.objects)
    ds = {'train': train, 'val': val}[args.split]
    tri_bank = build_triplet_bank(ds)            # key -> (de_i, ef_i)
    fd_bank = _subset_keys(ds['fd'])             # key -> fd_i
    gen_set = set(args.gen_eflesh)

    # Per object: pick one episode (seeded INDEPENDENTLY per object, so locking
    # one object's seed never disturbs the others), then the PEAK-CONTACT frame
    # within it, so every shown sample is strictly at maximum contact (never
    # approach or release). Objects in --gen_eflesh use the fd pair (real D+F,
    # generated eFlesh); the rest use the real de∩ef triplet.
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(seeds) == 1:
        seeds = seeds * len(args.objects)
    if len(seeds) != len(args.objects):
        ap.error(f"--seeds expects 1 or {len(args.objects)} values, got {len(seeds)}")

    chosen = {}
    for obj, sd in zip(args.objects, seeds):
        mode = 'fd' if obj in gen_set else 'tri'
        bank = fd_bank if mode == 'fd' else tri_bank
        keys = [k for k in bank if k[0] == obj]
        if not keys:
            print(f"  WARNING no {mode} sample found for '{obj}' — block left blank")
            chosen[obj] = None
            continue
        eps = {}
        for k in keys:
            eps.setdefault(k[1], []).append(k)
        ep = sorted(eps)[np.random.RandomState(sd).randint(len(eps))]
        frames = sorted(eps[ep], key=lambda k: k[2])
        flexi_idx = (lambda k: bank[k]) if mode == 'fd' else (lambda k: bank[k][1])
        scores = [contact_score(_flexi_for(ds, mode, flexi_idx(k))) for k in frames]
        best = int(np.argmax(scores))
        key = frames[best]
        chosen[obj] = (mode, bank[key])
        print(f"  {obj}: mode={mode} seed={sd} episode {ep} — peak-contact frame "
              f"(obj,ep,fr)={key}  score={scores[best]:.3f}  "
              f"(scanned {len(frames)} frames)")

    # ── render one 3x4 block (works on a Figure or a SubFigure) ──
    col_titles = ['GT', 'Self', 'Cross', 'Cross']

    def draw_block(container, obj):
        container.suptitle(obj, fontsize=14, fontweight='bold')
        axes = container.subplots(3, 4)
        sel = chosen[obj]
        if sel is None:
            for ax in axes.ravel():
                ax.set_xticks([]); ax.set_yticks([])
            return
        mode, idx = sel
        # gt per sensor (eFlesh GT is generated in 'fd' mode) + latents (z == mu)
        gt, z = load_block(ds, model, args.device, mode, idx)
        for r, s in enumerate(ORDER):
            others = [o for o in ORDER if o != s]
            cells = [
                (gt[s], f'{s} GT'),                                  # GT
                (model._decoder_for(s)(z[s]), f'{s}←{s}'),     # self
                (model._decoder_for(s)(z[others[0]]), f'{s}←{others[0]}'),
                (model._decoder_for(s)(z[others[1]]), f'{s}←{others[1]}'),
            ]
            vrange = value_range(s, [img for img, _ in cells])
            for c, (img, title) in enumerate(cells):
                ax = axes[r, c]
                disp(ax, s, img, vrange)
                ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel(f'{s}\n{SENSOR_NAME[s]}', fontsize=9,
                                  rotation=0, ha='right', va='center', labelpad=18)
                if r == 0:
                    ax.text(0.5, 1.28, col_titles[c], transform=ax.transAxes,
                            ha='center', va='bottom', fontsize=9, color='dimgray')

    # ── combined figure: all blocks side by side ──
    n = len(args.objects)
    fig = plt.figure(figsize=(4.2 * n, 4.6), constrained_layout=True)
    subfigs = np.atleast_1d(fig.subfigures(1, n, wspace=0.06))
    for sf, obj in zip(subfigs, args.objects):
        draw_block(sf, obj)
    fig.suptitle(f'Self- & cross-modal reconstruction (0_TACTX, {args.split}) — '
                 'rows: D / E / F   cols: GT | self | cross | cross',
                 fontsize=12, y=1.06)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved combined  {out_path}")

    # ── per-object blocks saved individually (assemble the best ones later) ──
    base, ext = os.path.splitext(out_path)
    for obj in args.objects:
        f = plt.figure(figsize=(4.2, 4.8), constrained_layout=True)
        draw_block(f, obj)
        p = f"{base}_{obj}{ext}"
        f.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(f)
        print(f"Saved block     {p}")


if __name__ == '__main__':
    main()
