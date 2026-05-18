"""
Visualize teleop data (latent_space_teleop) in the latent space of a
cross_modal_vae3 checkpoint (default: scratch11_shear_efs3).

Each teleop .pkl frame is {ts, s0, s1, [img0, img1]}; s0 = LEFT sensor,
s1 = RIGHT sensor. Three combos:
    left_daimon_right_eflesh  -> de pair  (daimon=s0, eflesh=s1)
    left_eflesh_right_flexi   -> ef pair  (eflesh=s0, flexi=s1)
    left_daimon_right_flexi   -> fd pair  (daimon=s0, flexi=s1)

PIPELINE per frame
------------------
1. raw -> processed-format transform (the per-channel recipe that built
   latent_data_processed_static):

     Daimon  (240,320,5):  depth   = ch0, reshape (240,320)
                           shear_x = ch3, reshape (240,320)
                           shear_y = ch4, reshape (240,320)
                           -> stack [depth, shear_x, shear_y] = (3,240,320)
                           fliplr depth & shear_y; shear_x never.
     eFlesh  (5,3):        reshape(-1) -> (15,).  never mirrored.
     FlexiTac(12,16):      astype float32; fliplr the 16-axis per pair rule.

   Per-pair mirroring (resolved with each combo's left/right assignment):
     de daimon (left)  : fliplr depth+shear_y      -> YES
     fd daimon (left)  : fliplr depth+shear_y      -> YES (daimon on left)
     ef flexi  (right) : fliplr 16-axis            -> YES (ef: always)
     fd flexi  (right) : fliplr 16-axis            -> NO  (flexi not on left)
     eflesh            : never

2. eval transform = rot180, matching dataset_sachin_static's per-pair
   extra_transforms:  de->daimon, ef->flexi, fd->daimon  (eflesh/fd-flexi: no).

3. resize Daimon to image_size (224); z-score every sensor with
   dataset_sachin_static stats.

4. encode with the checkpoint -> 16-d mu.

5. t-SNE of pooled latents, coloured by sensor (D/E/F), marker by source combo.

The ef combo is frame-subsampled per episode by --ef_stride (default 3) to
match the ef_stride-3 training of scratch11_shear_efs3.

Usage:
    python scripts/plot_teleop_latents.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_static_run_scratch11_shear_efs3_20260517_043942 \
        --device cuda:0
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torchvision.transforms.functional as TF
import yaml
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.utils.models_setup import get_model


_TELEOP_ROOT = 'tactile_benchmark/data/latent_space_teleop/pnp'

# Daimon raw (240,320,5): which channels are depth / shear_x / shear_y.
_DAIMON_DEPTH, _DAIMON_SX, _DAIMON_SY = 0, 3, 4

# Resolved per-combo spec. For each combo, each slot (s0=left, s1=right):
#   sensor       : 'D' / 'E' / 'F' (which encoder)
#   daimon_fliplr: fliplr depth & shear_y (Daimon only)
#   flexi_fliplr : fliplr the 16-axis    (FlexiTac only)
#   rot180       : eval-time rot180 (dataset_sachin_static extra_transforms)
_COMBO_SPEC = {
    'left_daimon_right_eflesh': {            # de pair
        's0': dict(sensor='D', daimon_fliplr=True,  rot180=True),
        's1': dict(sensor='E',                      rot180=False),
    },
    'left_eflesh_right_flexi': {             # ef pair
        's0': dict(sensor='E',                      rot180=False),
        's1': dict(sensor='F', flexi_fliplr=True,   rot180=True),
    },
    'left_daimon_right_flexi': {             # fd pair
        's0': dict(sensor='D', daimon_fliplr=True,  rot180=True),
        's1': dict(sensor='F', flexi_fliplr=False,  rot180=False),
    },
}

_SENSOR_COLOR = {'D': '#2171b5', 'E': '#e6261f', 'F': '#2ca02c'}
_COMBO_MARKER = {'left_daimon_right_eflesh': 'o',
                 'left_eflesh_right_flexi': '^',
                 'left_daimon_right_flexi': 's'}


# ---------------------------------------------------------------------------
# Step 1+2 — raw -> processed-format transform, then rot180
# ---------------------------------------------------------------------------

def _rot180(a):
    """Flip the last two axes (rot180 for both (C,H,W) and (H,W) arrays)."""
    return np.flip(a, axis=(-2, -1))


def transform_daimon(s_raw, daimon_fliplr, rot180):
    """Daimon raw (240,320,5) -> (3,240,320) = [depth, shear_x, shear_y]."""
    s = np.asarray(s_raw, dtype=np.float32)
    depth = s[:, :, _DAIMON_DEPTH]
    shear_x = s[:, :, _DAIMON_SX]
    shear_y = s[:, :, _DAIMON_SY]
    if daimon_fliplr:
        depth = np.fliplr(depth)
        shear_y = np.fliplr(shear_y)        # shear_x: unchanged
    out = np.stack([depth, shear_x, shear_y], axis=0)   # (3,240,320)
    if rot180:
        out = _rot180(out)
    return np.ascontiguousarray(out)


def transform_eflesh(s_raw, rot180):
    """eFlesh raw (5,3) -> (15,). Never mirrored / rotated (rot180 ignored)."""
    return np.ascontiguousarray(
        np.asarray(s_raw, dtype=np.float32).reshape(-1))


def transform_flexi(s_raw, flexi_fliplr, rot180):
    """FlexiTac raw (12,16) -> (12,16) float32."""
    s = np.asarray(s_raw, dtype=np.float32)
    if flexi_fliplr:
        s = np.fliplr(s)
    if rot180:
        s = _rot180(s)
    return np.ascontiguousarray(s)


def apply_transform(s_raw, spec):
    """Dispatch the step-1+2 transform by sensor type."""
    if spec['sensor'] == 'D':
        return transform_daimon(s_raw, spec.get('daimon_fliplr', False),
                                spec['rot180'])
    if spec['sensor'] == 'E':
        return transform_eflesh(s_raw, spec['rot180'])
    return transform_flexi(s_raw, spec.get('flexi_fliplr', False),
                           spec['rot180'])


# ---------------------------------------------------------------------------
# Step 3 — resize + z-score for the model (dataset_sachin_static stats)
# ---------------------------------------------------------------------------

def _zscore(x, mean, std):
    """x: (B, ...) tensor. mean/std broadcast over the channel/feature dim."""
    m = torch.as_tensor(mean, dtype=torch.float32)
    s = torch.as_tensor(std, dtype=torch.float32)
    if x.dim() == 4 and m.numel() == x.shape[1]:           # (B,C,H,W)
        m, s = m.view(1, -1, 1, 1), s.view(1, -1, 1, 1)
    elif x.dim() == 2 and m.numel() == x.shape[1]:         # (B,D)
        m, s = m.view(1, -1), s.view(1, -1)
    # scalar stats (flexi) broadcast as-is
    return (x - m) / (s + 1e-8)


def preprocess_batch(arrs, sensor, stats, image_size):
    """Stack transformed arrays -> z-scored model-ready tensor."""
    x = torch.from_numpy(np.stack(arrs, axis=0)).float()
    if sensor == 'D':                                      # (B,3,240,320)
        x = TF.resize(x, [image_size, image_size], antialias=True)
        x = _zscore(x, stats['daimon']['mean'], stats['daimon']['std'])
    elif sensor == 'E':                                    # (B,15)
        x = _zscore(x, stats['eflesh']['mean'], stats['eflesh']['std'])
    else:                                                  # (B,12,16)
        fm, fs = stats['flexi']['mean'], stats['flexi']['std']
        fm = fm[0] if isinstance(fm, (list, tuple)) else float(fm)
        fs = fs[0] if isinstance(fs, (list, tuple)) else float(fs)
        x = (x - fm) / (fs + 1e-8)
    return x


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------

def _episode_dirs(combo_dir):
    return sorted(d for d in glob.glob(os.path.join(combo_dir, 'tactile',
                                                    'episode_*'))
                  if os.path.isdir(d))


def _frame_files(ep_dir):
    """<timestamp>.pkl frames, time-ordered; metadata.pkl excluded."""
    fs = []
    for f in glob.glob(os.path.join(ep_dir, '*.pkl')):
        try:
            float(os.path.splitext(os.path.basename(f))[0])
        except ValueError:
            continue
        fs.append(f)
    return sorted(fs)


def contact_frames(frame_paths, threshold):
    """Crop one episode's frames to the contact frames.

    Per frame, each sensor's activity = std of its RAW reading; each sensor's
    activity series is normalized to [0,1] within the episode; a frame is
    'contact' if either sensor exceeds `threshold`. Detection is on the raw
    reading — the transform table only slices/flips, so it does not change
    whether a frame is in contact. Returns the kept frame paths (time order).
    """
    if not frame_paths:
        return []
    a0, a1 = [], []
    for p in frame_paths:
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
        a0.append(float(np.asarray(d['s0'], dtype=np.float32).std()))
        a1.append(float(np.asarray(d['s1'], dtype=np.float32).std()))

    def _norm(a):
        a = np.asarray(a, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)

    keep = (_norm(a0) > threshold) | (_norm(a1) > threshold)
    return [p for p, k in zip(frame_paths, keep) if k]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_combo(combo, teleop_root, model, stats, image_size, device,
                 ef_stride, batch_size, max_points_per_sensor, seed,
                 contact_threshold, all_frames):
    """Encode one combo's teleop frames. Returns {('D'/'E'/'F'): mu_array}
    contributed by this combo (s0 and s1 are different sensors).

    Frames are contact-cropped (unless all_frames) BEFORE encoding, then the
    ef combo is ef_stride-subsampled."""
    spec = _COMBO_SPEC[combo]
    combo_dir = os.path.join(teleop_root, combo)
    if not os.path.isdir(combo_dir):
        print(f"  [skip] missing {combo_dir}")
        return {}

    is_ef = (combo == 'left_eflesh_right_flexi')
    stride = ef_stride if is_ef else 1

    # collect transformed arrays per slot
    buf = {'s0': [], 's1': []}
    n_raw = n_contact = 0
    for ep in _episode_dirs(combo_dir):
        frames = _frame_files(ep)
        n_raw += len(frames)
        if not all_frames:
            frames = contact_frames(frames, contact_threshold)   # crop to contact
        n_contact += len(frames)
        frames = frames[::stride]                                # then ef-stride
        for fp in frames:
            with open(fp, 'rb') as fh:
                d = pickle.load(fh)
            buf['s0'].append(apply_transform(d['s0'], spec['s0']))
            buf['s1'].append(apply_transform(d['s1'], spec['s1']))

    if not all_frames:
        print(f"  contact crop: {n_contact}/{n_raw} frames kept")

    out = {}
    for slot in ('s0', 's1'):
        arrs = buf[slot]
        sensor = spec[slot]['sensor']
        if not arrs:
            continue
        # subsample for t-SNE budget (deterministic)
        if len(arrs) > max_points_per_sensor:
            rng = np.random.default_rng(seed)
            idx = sorted(rng.choice(len(arrs), max_points_per_sensor,
                                    replace=False).tolist())
            arrs = [arrs[i] for i in idx]
        mus = []
        for s in range(0, len(arrs), batch_size):
            x = preprocess_batch(arrs[s:s + batch_size], sensor, stats,
                                 image_size).to(device)
            _, mu, _, _ = model._enc(x, sensor)
            mus.append(mu.cpu().numpy())
        mu = np.concatenate(mus, axis=0).astype(np.float32)
        out[slot] = (sensor, mu)
        print(f"  {combo} {slot} ({sensor}): {mu.shape}")
    return out


def _mmd_rbf(x, y, bandwidths=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """Multi-kernel RBF MMD² between two latent sample sets (numpy in)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    out = 0.0
    for s in bandwidths:
        out += (torch.exp(-xx / (2 * s)).mean()
                + torch.exp(-yy / (2 * s)).mean()
                - 2 * torch.exp(-xy / (2 * s)).mean()).item()
    return out / len(bandwidths)


def _report_distances(combo_res, groups):
    """Print latent-space distances.

    1. Per-combo MATCHED-pair distance: s0[i] and s1[i] are the same contact
       frame seen by two sensors — for a working cross-modal latent their
       latents should coincide (L2 -> 0, cosine -> 1).
    2. Pooled per-sensor distribution distance: MMD² and centroid L2 between
       the D / E / F latent clouds.
    """
    print('\n' + '=' * 60)
    print('Latent-space distances')
    print('=' * 60)
    print('Per-combo matched-pair distance (s0 vs s1, same contact frame):')
    for combo, res in combo_res.items():
        if 's0' not in res or 's1' not in res:
            continue
        (sa, mu_a), (sb, mu_b) = res['s0'], res['s1']
        n = min(len(mu_a), len(mu_b))
        a, b = mu_a[:n], mu_b[:n]
        l2 = float(np.linalg.norm(a - b, axis=1).mean())
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        cos = float((an * bn).sum(axis=1).mean())
        print(f'  {combo:26s} ({sa} vs {sb}): '
              f'L2={l2:.4f}  cos={cos:+.4f}  (n={n})')

    # pooled per-sensor
    pool = {}
    for sensor, _combo, mu in groups:
        pool.setdefault(sensor, []).append(mu)
    pool = {s: np.concatenate(v, axis=0) for s, v in pool.items()}
    print('\nPooled per-sensor distribution distance:')
    for a, b in (('D', 'E'), ('E', 'F'), ('D', 'F')):
        if a in pool and b in pool:
            mmd = _mmd_rbf(pool[a], pool[b])
            cen = float(np.linalg.norm(pool[a].mean(0) - pool[b].mean(0)))
            print(f'  {a} vs {b}:  MMD²={mmd:.5f}   centroid-L2={cen:.4f}')
    print('=' * 60)


def _load_model(checkpoint_dir, weights, device):
    with open(os.path.join(checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model('cross_modal_vae3', model_details).to(device)
    ckpt = torch.load(os.path.join(checkpoint_dir, weights),
                      map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True,
                        help='cross_modal_vae3 checkpoint dir '
                             '(e.g. ..._scratch11_shear_efs3_<TS>).')
    parser.add_argument('--weights', default='model.pt',
                        help="Weights file in the checkpoint dir "
                             "(e.g. model_ep300.pt). Default model.pt.")
    parser.add_argument('--teleop_root', default=_TELEOP_ROOT)
    parser.add_argument('--stats_dataset', default='dataset_sachin_static',
                        help='datasets_config_sachin.yaml entry for the '
                             'z-score stats (match the training run).')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ef_stride', type=int, default=3,
                        help='Per-episode frame stride for the ef combo '
                             '(match the training ef_stride). Default 3.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_points', type=int, default=1500,
                        help='Cap per (combo, slot) fed to t-SNE.')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--contact_threshold', type=float, default=0.12,
                        help='Contact-crop threshold on per-sensor normalized '
                             'activity [0,1]. Frames below it (no contact) are '
                             'dropped before encoding.')
    parser.add_argument('--all_frames', action='store_true',
                        help='Disable contact cropping — encode every frame.')
    parser.add_argument('--out', default='teleop_latents.png')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    teleop_root = (args.teleop_root if os.path.isabs(args.teleop_root)
                   else os.path.join(repo_root, args.teleop_root))

    # z-score stats from the sachin config (must match the training run).
    with open(os.path.join(repo_root, 'config',
                           'datasets_config_sachin.yaml')) as f:
        dscfg = yaml.safe_load(f)[args.stats_dataset]
    stats = {'daimon': dscfg['daimon_stats'],
             'eflesh': dscfg['eflesh_stats'],
             'flexi': dscfg['flexitac_stats']}
    image_size = dscfg.get('image_size', 224)

    model = _load_model(args.checkpoint_dir, args.weights, args.device)
    print(f"Loaded {args.checkpoint_dir}/{args.weights}")

    # Encode every combo; pool latents per sensor, tracking source combo.
    groups = []                       # list of (sensor, combo, mu_array)
    combo_res = {}                    # combo -> {'s0': (sensor,mu), 's1': ...}
    for combo in _COMBO_SPEC:
        print(f"=== {combo} ===")
        res = encode_combo(combo, teleop_root, model, stats, image_size,
                           args.device, args.ef_stride, args.batch_size,
                           args.max_points, args.seed,
                           args.contact_threshold, args.all_frames)
        combo_res[combo] = res
        for slot, (sensor, mu) in res.items():
            groups.append((sensor, combo, mu))

    if not groups:
        print("No teleop data encoded — check --teleop_root.")
        return

    _report_distances(combo_res, groups)

    # pooled t-SNE
    sizes = [g[2].shape[0] for g in groups]
    pooled = np.concatenate([g[2] for g in groups], axis=0)
    perp = min(args.tsne_perplexity, max(5, (len(pooled) - 1) // 3))
    emb = TSNE(n_components=2, perplexity=perp, init='pca',
               random_state=args.seed).fit_transform(pooled)

    fig, ax = plt.subplots(figsize=(9, 8))
    off = 0
    for (sensor, combo, mu), sz in zip(groups, sizes):
        pts = emb[off:off + sz]
        off += sz
        ax.scatter(pts[:, 0], pts[:, 1], s=9, alpha=0.55,
                   c=_SENSOR_COLOR[sensor], marker=_COMBO_MARKER[combo],
                   edgecolors='none')

    handles = [Line2D([], [], color=c, marker='o', ls='', label=f'{s}')
               for s, c in _SENSOR_COLOR.items()]
    handles += [Line2D([], [], color='gray', marker=m, ls='', label=cb)
                for cb, m in _COMBO_MARKER.items()]
    ax.legend(handles=handles, fontsize=7, loc='best', ncol=2, framealpha=0.7)
    ax.set_title(f"Teleop latents — {os.path.basename(args.checkpoint_dir.rstrip('/'))}\n"
                 f"colour = sensor (D/E/F), marker = source combo",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
