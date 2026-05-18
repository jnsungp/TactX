"""
Self + cross reconstruction VIDEO for a processed sachin dataset (.pt format).

Per pair (de/ef/fd), per episode, writes a GIF where each frame is a 2x3 grid:
    sensor_a:  GT | self  dec_a(z_a) | cross dec_a(z_b)
    sensor_b:  GT | self  dec_b(z_b) | cross dec_b(z_a)

The original pair (column 1) plays alongside its self-recon (col 2) and
cross-recon (col 3), frame by frame over the episode.

Model AND z-score stats come from --checkpoint_dir: stats are read from the
checkpoint's saved dataset_details.yaml, i.e. exactly what the model was
trained with (so e.g. plain_cyl is normalized with its cylinder-era stats,
regardless of which --dataset_name supplies the data).

Data layout (data_root, de/ef/fd pairs, rot180) comes from the --dataset_name
entry in datasets_config_sachin.yaml; --objects selects object folders.

Output: visualizations/recon_video_sachin/<dataset>__<obj>__<pair>__epNNNN.gif

Usage:
    # dynamic pnp:
    python scripts/viz_recon_video_sachin.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_dynalign_plain_cyl_20260511_214626 \
        --dataset_name dataset_sachin_dynamic_pnp --objects pnp --device cuda:0
    # static cylinder:
    python scripts/viz_recon_video_sachin.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_dynalign_plain_cyl_20260511_214626 \
        --dataset_name dataset_sachin_static --objects cylinder --device cuda:0
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_sachin_pair_similarity import _load_model, _build_pair_dataset
from viz_teleop_recon import _vis            # per-sensor single-sample renderer

_NAME2LETTER = {'daimon': 'D', 'eflesh': 'E', 'flexi': 'F'}


def _fig_to_pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(rgba[..., :3], mode='RGB')


@torch.no_grad()
def _episode_recons(ds, idx, pair_key, model, device, batch_size):
    """Run forward_pair over one episode's samples (idx, frame-ordered).
    Returns dict of CPU tensors: x_a,x_b, self_a,cross_a, self_b,cross_b."""
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False)
    acc = defaultdict(list)
    for (xa, xb), _labels in loader:
        out = model.forward_pair(xa.to(device), xb.to(device), pair_key)
        acc['x_a'].append(xa)
        acc['x_b'].append(xb)
        acc['self_a'].append(out['recon_aa'].cpu())   # dec_a(z_a)
        acc['cross_a'].append(out['recon_ba'].cpu())   # dec_a(z_b)
        acc['self_b'].append(out['recon_bb'].cpu())    # dec_b(z_b)
        acc['cross_b'].append(out['recon_ab'].cpu())   # dec_b(z_a)
    return {k: torch.cat(v, dim=0) for k, v in acc.items()}


def render_episode_gif(rec, sa, sb, title, out_path, fps):
    col_titles = ['original', 'self  dec(z_self)', 'cross dec(z_other)']
    n = len(rec['x_a'])
    pil_frames = []
    for i in range(n):
        fig, axs = plt.subplots(2, 3, figsize=(10.5, 7), dpi=90)
        rows = [
            (sa, [rec['x_a'][i], rec['self_a'][i], rec['cross_a'][i]]),
            (sb, [rec['x_b'][i], rec['self_b'][i], rec['cross_b'][i]]),
        ]
        for r, (sensor, panels) in enumerate(rows):
            for c, panel in enumerate(panels):
                _vis(axs[r][c], sensor, panel)
                if r == 0:
                    axs[r][c].set_title(col_titles[c], fontsize=9)
                if c == 0:
                    axs[r][c].set_ylabel(sensor, fontsize=10, rotation=0,
                                         ha='right', va='center', labelpad=20)
        fig.suptitle(f'{title} | frame {i:04d}', fontsize=10)
        fig.tight_layout(rect=[0.03, 0, 1, 0.96])
        pil_frames.append(_fig_to_pil(fig))
        plt.close(fig)
    duration_ms = max(int(1000 / max(fps, 1)), 20)
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=duration_ms, loop=0, disposal=2)
    print(f'  wrote {out_path}  ({n} frames @ {fps} fps)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--weights', default='model.pt')
    parser.add_argument('--dataset_name', required=True,
                        help='datasets_config_sachin.yaml entry — supplies the '
                             'data layout (data_root, pairs). Its stats are '
                             'IGNORED; checkpoint stats are used instead.')
    parser.add_argument('--objects', nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pairs', nargs='+', default=['de', 'ef', 'fd'],
                        choices=['de', 'ef', 'fd'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--max_frames', type=int, default=250,
                        help='Cap on frames per GIF (uniform subsample).')
    parser.add_argument('--max_episodes', type=int, default=5,
                        help='Episodes to render per pair (0 = all).')
    parser.add_argument('--out_dir', default='visualizations/recon_video_sachin')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = (args.out_dir if os.path.isabs(args.out_dir)
               else os.path.join(repo_root, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # data layout from the config entry
    with open(os.path.join(repo_root, 'config',
                           'datasets_config_sachin.yaml')) as f:
        dd = yaml.safe_load(f)[args.dataset_name]
    # stats from the CHECKPOINT (what the model was trained with)
    with open(os.path.join(args.checkpoint_dir, 'dataset_details.yaml')) as f:
        ckpt_dd = yaml.safe_load(f)
    for k in ('daimon_stats', 'eflesh_stats', 'flexitac_stats'):
        if k in ckpt_dd:
            dd[k] = ckpt_dd[k]
    print(f"stats: from checkpoint ({args.checkpoint_dir})")

    data_root = dd['data_root']
    if not os.path.isabs(data_root):
        data_root = os.path.join(repo_root, data_root)
    image_size = dd.get('image_size', 224)
    pairs = dd['pairs']

    model = _load_model(args.checkpoint_dir, args.weights, args.device)
    print(f"Loaded {args.checkpoint_dir}/{args.weights}\n")

    for pk in args.pairs:
        if pk not in pairs:
            continue
        # ef_stride=1: keep every frame so the video plays smooth
        ds, sa_name, sb_name = _build_pair_dataset(
            dd, data_root, pk, pairs[pk], image_size, args.objects, ef_stride=1)
        sa, sb = _NAME2LETTER[sa_name], _NAME2LETTER[sb_name]
        if len(ds) == 0:
            print(f'[{pk}] no samples')
            continue

        # group sample indices by (object, episode), frame-ordered
        groups = defaultdict(list)
        for i in range(len(ds)):
            groups[(ds.object_per_sample[i], ds.episode_per_sample[i])].append(i)
        items = sorted(groups.items())
        if args.max_episodes > 0:
            items = items[:args.max_episodes]
        print(f'[{pk}] {sa_name}+{sb_name}: {len(groups)} episodes, '
              f'rendering {len(items)}')

        for (obj_id, ep_id), idx in items:
            idx.sort(key=lambda i: ds.frame_per_sample[i])
            if args.max_frames and len(idx) > args.max_frames:
                sel = np.linspace(0, len(idx) - 1, args.max_frames).astype(int)
                idx = [idx[s] for s in sel]
            obj_name = (ds.object_names[obj_id]
                        if obj_id < len(ds.object_names) else f'o{obj_id}')
            rec = _episode_recons(ds, idx, pk, model, args.device,
                                  args.batch_size)
            title = f'{args.dataset_name} | {obj_name} | {pk} ({sa}-{sb}) | ep{ep_id:04d}'
            out = os.path.join(
                out_dir,
                f'{args.dataset_name}__{obj_name}__{pk}__ep{ep_id:04d}.gif')
            render_episode_gif(rec, sa, sb, title, out, args.fps)

    print(f'\nGIFs in {out_dir}/')


if __name__ == '__main__':
    main()
