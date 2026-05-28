"""
Side-by-side GIF per episode for a paired sensor folder.

For each episode under
    <data_root>/<pair>/{<sensor_a>,<sensor_b>}/<object>/ep<NNNN>_fr<NNNN>.pt
this writes a single animated GIF where the two sensors play synchronously
(frame i of sensor_a beside frame i of sensor_b), saved to
    <out_dir>/<pair>/<object>/ep<NNNN>.gif

Daimon (3,240,320) is shown as the depth channel (ch0; duplicated in the
preprocess so any channel works) colorized with HOT. Flexi (12,16) is
nearest-upsampled to match daimon's height and colorized with VIRIDIS so
the two panels are visually distinct. eFlesh (15-vec) is rendered as a bar
chart panel (used only if you point this at de_pair / ef_pair).

Each frame carries a small label ("Daimon (depth)" / "Flexi (12x16)") and a
frame counter so you can read the timing.

Per-episode min-max normalization keeps each ep's dynamic range readable
without one big contact saturating others.

Usage (default = fd_pair / small_ball):
  python scripts/viz_pair_gifs.py
  python scripts/viz_pair_gifs.py --pair fd_pair --object small_ball
  python scripts/viz_pair_gifs.py --episodes 1 3 7   # only those eps
  python scripts/viz_pair_gifs.py --fps 15
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


PAIR_SENSORS = {
    "de_pair": ("daimon", "eflesh"),
    "ef_pair": ("eflesh", "flexi"),
    "fd_pair": ("daimon", "flexi"),
}

PANEL_H, PANEL_W = 240, 320
LABEL_H = 22
COUNTER_H = 22


# ---- per-sensor frame renderers -------------------------------------------

def _norm01(stack):
    """Per-episode min-max normalize a stacked (T, ...) tensor to [0,1]."""
    lo, hi = float(stack.min()), float(stack.max())
    if hi - lo < 1e-8:
        return np.zeros_like(stack, dtype=np.float32)
    return ((stack - lo) / (hi - lo)).astype(np.float32)


def _colormap_rgb(u8_2d, cmap):
    bgr = cv2.applyColorMap(u8_2d, cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_daimon(arrs):
    """(T,3,H,W) -> list of (PANEL_H,PANEL_W,3) uint8 RGB via HOT colormap."""
    stack = np.stack([a[0] for a in arrs], axis=0)   # ch0 == depth
    norm = _norm01(stack)
    out = []
    for fr in norm:
        u8 = (fr * 255.0).astype(np.uint8)
        rgb = _colormap_rgb(u8, cv2.COLORMAP_HOT)
        if rgb.shape[:2] != (PANEL_H, PANEL_W):
            rgb = cv2.resize(rgb, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        out.append(rgb)
    return out


def render_flexi(arrs):
    """(T,12,16) -> list of (PANEL_H,PANEL_W,3) uint8 RGB via VIRIDIS, nearest upsample."""
    stack = np.stack(arrs, axis=0)
    norm = _norm01(stack)
    out = []
    for fr in norm:
        u8 = (fr * 255.0).astype(np.uint8)
        rgb = _colormap_rgb(u8, cv2.COLORMAP_VIRIDIS)
        rgb = cv2.resize(rgb, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)
        out.append(rgb)
    return out


def render_eflesh(arrs):
    """(T,15) -> list of (PANEL_H,PANEL_W,3) uint8 bar-chart panels."""
    stack = np.stack(arrs, axis=0)
    # symmetric per-episode normalize so negative/positive both visible
    m = float(np.abs(stack).max())
    if m < 1e-8:
        m = 1.0
    bar_w = PANEL_W // 15
    midline = PANEL_H // 2
    out = []
    for fr in stack:
        canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        canvas[midline - 1:midline + 1, :] = (90, 90, 90)
        for i, v in enumerate(fr):
            h = int((v / m) * (PANEL_H * 0.45))
            x0 = i * bar_w + 2
            x1 = (i + 1) * bar_w - 2
            color = (60, 160, 255) if v >= 0 else (255, 110, 110)
            y0, y1 = (midline - h, midline) if h >= 0 else (midline, midline - h)
            y0, y1 = max(0, y0), min(PANEL_H, y1)
            canvas[y0:y1, x0:x1] = color
        out.append(canvas)
    return out


SENSOR_RENDERER = {"daimon": render_daimon, "flexi": render_flexi, "eflesh": render_eflesh}


# ---- labels / counter ------------------------------------------------------

def _strip(text, width, height, bg=(30, 30, 30), fg=(235, 235, 235), font_scale=0.6):
    strip = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(strip, text, (8, height - 6), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg, 1, cv2.LINE_AA)
    return strip


def build_panel(rgb, label):
    return np.vstack([_strip(label, PANEL_W, LABEL_H), rgb])


def add_counter(combined, text):
    return np.vstack([combined, _strip(text, combined.shape[1], COUNTER_H,
                                       bg=(20, 20, 20))])


# ---- IO --------------------------------------------------------------------

_FR_RE = re.compile(r"ep(\d+)_fr(\d+)\.pt$")


def load_episode_frames(sensor_dir, ep):
    files = []
    for f in glob.glob(os.path.join(sensor_dir, f"ep{ep:04d}_fr*.pt")):
        m = _FR_RE.search(os.path.basename(f))
        if m:
            files.append((int(m.group(2)), f))
    files.sort()
    return [torch.load(p, map_location="cpu", weights_only=False).numpy()
            for _, p in files]


def list_episodes(sensor_dir):
    eps = set()
    for f in glob.glob(os.path.join(sensor_dir, "ep*_fr*.pt")):
        m = _FR_RE.search(os.path.basename(f))
        if m:
            eps.add(int(m.group(1)))
    return sorted(eps)


def save_gif(frames, out_path, fps):
    duration_ms = int(round(1000.0 / max(1, fps)))
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0, optimize=False, disposal=2)


# ---- main ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/junsung/cross_modal_tactile/"
                   "tactile_benchmark/data/latent_data_processed_mirror")
    p.add_argument("--pair", choices=list(PAIR_SENSORS), default="fd_pair")
    p.add_argument("--object", default="small_ball")
    p.add_argument("--out_dir", default="/home/junsung/cross_modal_tactile/vis_latent")
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="Specific episode ids (default: all found).")
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    sa, sb = PAIR_SENSORS[args.pair]
    dir_a = os.path.join(args.data_root, args.pair, sa, args.object)
    dir_b = os.path.join(args.data_root, args.pair, sb, args.object)
    if not (os.path.isdir(dir_a) and os.path.isdir(dir_b)):
        raise FileNotFoundError(f"missing {dir_a} or {dir_b}")

    eps_a = list_episodes(dir_a)
    eps_b = list_episodes(dir_b)
    eps = sorted(set(eps_a) & set(eps_b))
    if args.episodes is not None:
        eps = [e for e in eps if e in set(args.episodes)]

    out_dir = os.path.join(args.out_dir, args.pair, args.object)
    os.makedirs(out_dir, exist_ok=True)

    label_a = f"{sa.capitalize()}" + (" (depth)" if sa == "daimon" else
                                       (" (12x16)" if sa == "flexi" else " (15)"))
    label_b = f"{sb.capitalize()}" + (" (depth)" if sb == "daimon" else
                                       (" (12x16)" if sb == "flexi" else " (15)"))
    render_a = SENSOR_RENDERER[sa]
    render_b = SENSOR_RENDERER[sb]

    print(f"[{args.pair}/{args.object}] {len(eps)} episodes -> {out_dir}")
    for ep in tqdm(eps, desc="episodes"):
        arrs_a = load_episode_frames(dir_a, ep)
        arrs_b = load_episode_frames(dir_b, ep)
        if not arrs_a or not arrs_b:
            continue
        viz_a = render_a(arrs_a)
        viz_b = render_b(arrs_b)
        T = min(len(viz_a), len(viz_b))
        out_frames = []
        for i in range(T):
            pa = build_panel(viz_a[i], label_a)
            pb = build_panel(viz_b[i], label_b)
            combined = np.hstack([pa, pb])
            combined = add_counter(
                combined,
                f"{args.pair} / {args.object} / ep{ep:04d}  fr {i:3d}/{T - 1}")
            out_frames.append(combined)
        save_gif(out_frames, os.path.join(out_dir, f"ep{ep:04d}.gif"), args.fps)

    print(f"\nDone. GIFs under {out_dir}/")


if __name__ == "__main__":
    main()
