"""
Video (GIF) of self + cross reconstruction for a teleop sensor pair, shown
together with the original pair — frame by frame over an episode.

Each video frame is a 2x3 grid:
    X GT | X self  dec_X(z_X) | X cross dec_X(z_Y)
    Y GT | Y self  dec_Y(z_Y) | Y cross dec_Y(z_X)

so the original pair (column 1) plays alongside its self-recon (column 2) and
cross-recon (column 3). One GIF per combo x episode, contact frames only.

Same encode pipeline as plot_teleop_latents.py (transform table + rot180 +
resize + z-score). Reuses the per-sensor renderer from viz_teleop_recon.py.

Output: visualizations/teleop_recon_video/<combo>__episode_XXXX.gif

Usage:
    python scripts/viz_teleop_recon_video.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_static_run_scratch11_shear_efs3_20260517_043942 \
        --device cuda:0
    python scripts/viz_teleop_recon_video.py --combos left_daimon_right_eflesh --n_episodes 1
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
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_teleop_latents import (
    apply_transform, contact_frames, preprocess_batch, _load_model,
    _COMBO_SPEC, _frame_files, _episode_dirs, _TELEOP_ROOT,
)
from viz_teleop_recon import _vis          # per-sensor single-sample renderer


@torch.no_grad()
def _encode_all(x, sensor, model, device, batch_size):
    """x: (N,...) CPU tensor -> mu (N,16) on device."""
    mus = []
    for s in range(0, len(x), batch_size):
        _, mu, _, _ = model._enc(x[s:s + batch_size].to(device), sensor)
        mus.append(mu)
    return torch.cat(mus, dim=0)


@torch.no_grad()
def _decode_all(dec, mu, batch_size):
    """mu: (N,16) device tensor -> recon (N,...) on CPU."""
    outs = []
    for s in range(0, len(mu), batch_size):
        outs.append(dec(mu[s:s + batch_size]).cpu())
    return torch.cat(outs, dim=0)


def _fig_to_pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(rgba[..., :3], mode='RGB')


@torch.no_grad()
def render_episode(combo, ep_dir, model, stats, image_size, device,
                   batch_size, fps, max_frames, contact_threshold,
                   all_frames, out_path):
    spec = _COMBO_SPEC[combo]
    sa, sb = spec['s0']['sensor'], spec['s1']['sensor']

    frames = _frame_files(ep_dir)
    if not all_frames:
        n0 = len(frames)
        frames = contact_frames(frames, contact_threshold)
        print(f'  {os.path.basename(ep_dir)}: contact {len(frames)}/{n0}')
    if not frames:
        print(f'  [skip] no frames {ep_dir}')
        return
    if max_frames and len(frames) > max_frames:
        sel = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in sel]

    # transform every frame -> model-ready tensors
    arr0, arr1 = [], []
    for fp in frames:
        with open(fp, 'rb') as fh:
            d = pickle.load(fh)
        arr0.append(apply_transform(d['s0'], spec['s0']))
        arr1.append(apply_transform(d['s1'], spec['s1']))
    x_a = preprocess_batch(arr0, sa, stats, image_size)        # CPU
    x_b = preprocess_batch(arr1, sb, stats, image_size)

    # encode + decode self / cross
    mu_a = _encode_all(x_a, sa, model, device, batch_size)
    mu_b = _encode_all(x_b, sb, model, device, batch_size)
    dec_a, dec_b = model._decoder_for(sa), model._decoder_for(sb)
    self_a = _decode_all(dec_a, mu_a, batch_size)              # dec_a(z_a)
    cross_a = _decode_all(dec_a, mu_b, batch_size)             # dec_a(z_b)
    self_b = _decode_all(dec_b, mu_b, batch_size)              # dec_b(z_b)
    cross_b = _decode_all(dec_b, mu_a, batch_size)             # dec_b(z_a)

    col_titles = ['original', 'self  dec(z_self)', 'cross dec(z_other)']
    pil_frames = []
    for i in range(len(frames)):
        fig, axs = plt.subplots(2, 3, figsize=(10.5, 7), dpi=90)
        rowsets = [
            (sa, [x_a[i], self_a[i], cross_a[i]]),
            (sb, [x_b[i], self_b[i], cross_b[i]]),
        ]
        for r, (sensor, panels) in enumerate(rowsets):
            for c, panel in enumerate(panels):
                _vis(axs[r][c], sensor, panel)
                if r == 0:
                    axs[r][c].set_title(col_titles[c], fontsize=9)
                if c == 0:
                    axs[r][c].set_ylabel(f'{sensor}', fontsize=10,
                                         rotation=0, ha='right', va='center',
                                         labelpad=20)
        fig.suptitle(f'{combo} | {os.path.basename(ep_dir)} | frame {i:04d}',
                     fontsize=10)
        fig.tight_layout(rect=[0.03, 0, 1, 0.96])
        pil_frames.append(_fig_to_pil(fig))
        plt.close(fig)

    duration_ms = max(int(1000 / max(fps, 1)), 20)
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=duration_ms, loop=0, disposal=2)
    print(f'  wrote {out_path}  ({len(pil_frames)} frames @ {fps} fps)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--weights', default='model.pt')
    parser.add_argument('--teleop_root', default=_TELEOP_ROOT)
    parser.add_argument('--stats_dataset', default='dataset_sachin_static')
    parser.add_argument('--combos', nargs='+', default=list(_COMBO_SPEC.keys()))
    parser.add_argument('--n_episodes', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--contact_threshold', type=float, default=0.12)
    parser.add_argument('--all_frames', action='store_true')
    parser.add_argument('--out_dir', default='visualizations/teleop_recon_video')
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

    for combo in args.combos:
        combo_dir = os.path.join(teleop_root, combo)
        if not os.path.isdir(combo_dir):
            print(f'[skip] no dir: {combo_dir}')
            continue
        eps = _episode_dirs(combo_dir)[:args.n_episodes]
        print(f'[{combo}]')
        for ep_dir in eps:
            out = os.path.join(out_dir,
                               f'{combo}__{os.path.basename(ep_dir)}.gif')
            render_episode(combo, ep_dir, model, stats, image_size,
                           args.device, args.batch_size, args.fps,
                           args.max_frames, args.contact_threshold,
                           args.all_frames, out)

    print(f'\nGIFs in {out_dir}/')


if __name__ == '__main__':
    main()
