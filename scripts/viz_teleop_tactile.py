"""Visualize 2 random tactile episodes from each left_*right_* pairing dir.

For each episode it renders an MP4 with the active tactile sensors shown
side-by-side, one video frame per recorded timestep.

Usage:
    python scripts/viz_teleop_tactile.py
    python scripts/viz_teleop_tactile.py --root <path/to/latent_space_teleop> --task pnp --n 2
"""
import argparse
import glob
import os
import pickle
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation


def load_frame(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def describe(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            describe(v, f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        print(f"  {prefix}: {type(obj).__name__} len={len(obj)}")
        if obj:
            describe(obj[0], f"{prefix}[0]")
    elif hasattr(obj, "shape"):
        print(f"  {prefix}: {type(obj).__name__} shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"  {prefix}: {type(obj).__name__} {repr(obj)[:80]}")


def find_image_arrays(obj, path=""):
    """Yield (path, ndarray) for arrays that look like images (H,W) or (H,W,C)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += find_image_arrays(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += find_image_arrays(v, f"{path}[{i}]")
    else:
        a = obj if isinstance(obj, np.ndarray) else None
        if a is None:
            try:
                a = np.asarray(obj)
            except Exception:
                a = None
        if isinstance(a, np.ndarray) and a.ndim in (2, 3) and np.issubdtype(
            a.dtype, np.number
        ):
            if a.ndim == 2 and min(a.shape) >= 8:
                out.append((path, a))
            elif a.ndim == 3 and a.shape[-1] in (1, 3, 4):
                out.append((path, a))
    return out


def to_displayable(a):
    a = np.asarray(a).astype(np.float32)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    return a


def episode_frames(ep_dir):
    return sorted(glob.glob(os.path.join(ep_dir, "[0-9]*.pkl")))


def select_active_keys(frames, probe=12):
    """Return keys whose pixels actually change across the episode."""
    idxs = np.linspace(0, len(frames) - 1, min(probe, len(frames))).astype(int)
    stacks = {}
    for i in idxs:
        for k, a in find_image_arrays(load_frame(frames[i])):
            stacks.setdefault(k, []).append(np.asarray(a, dtype=np.float32))
    active, dead = [], []
    for k, arrs in stacks.items():
        try:
            stk = np.stack(arrs, axis=0)
        except ValueError:
            active.append(k)  # variable shape -> treat as active
            continue
        # active if it varies over time AND is not an all-constant blank
        temporal_var = float(stk.std(axis=0).mean())
        spread = float(stk.max() - stk.min())
        (active if (temporal_var > 1e-6 and spread > 1e-6) else dead).append(k)
    return sorted(active), sorted(dead)


def viz_episode(ep_dir, out_path, fps=15, max_frames=400):
    frames = episode_frames(ep_dir)
    if not frames:
        print(f"  [skip] no frames in {ep_dir}")
        return
    print(f"  episode: {ep_dir}  ({len(frames)} frames)")
    describe(load_frame(frames[0]), "frame")

    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idxs]

    active, dead = select_active_keys(frames)
    print(f"  active sensors: {active}")
    if dead:
        print(f"  inactive (skipped): {dead}")
    if not active:
        print(f"  [skip] no active sensors in {ep_dir}")
        return

    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    axes = axes[0]
    ims = []
    first = load_frame(frames[0])
    arr0 = dict(find_image_arrays(first))
    for ax, k in zip(axes, active):
        ax.set_title(k, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(to_displayable(arr0[k]))
        ims.append(im)
    title = fig.suptitle("", fontsize=9)

    def update(fi):
        d = dict(find_image_arrays(load_frame(frames[fi])))
        for im, k in zip(ims, active):
            if k in d:
                im.set_data(to_displayable(d[k]))
        title.set_text(f"{os.path.relpath(ep_dir)}   frame {fi + 1}/{len(frames)}")
        return ims + [title]

    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )
    try:
        anim.save(out_path, writer="ffmpeg", fps=fps, dpi=110)
    except Exception as e:
        gif = os.path.splitext(out_path)[0] + ".gif"
        print(f"  ffmpeg failed ({e}); writing GIF instead")
        anim.save(gif, writer="pillow", fps=fps)
        out_path = gif
    plt.close(fig)
    print(f"  -> saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.join(
        here, "..", "tactile_benchmark", "data", "latent_space_teleop"
    )
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--task", default="pnp")
    ap.add_argument("--n", type=int, default=2, help="episodes per pairing")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    task_dir = os.path.join(args.root, args.task)
    out_dir = args.out or os.path.join(task_dir, "_viz")
    os.makedirs(out_dir, exist_ok=True)

    pairings = sorted(
        d
        for d in os.listdir(task_dir)
        if d.startswith("left_") and os.path.isdir(os.path.join(task_dir, d))
    )
    print(f"pairings: {pairings}")

    for pairing in pairings:
        tac_dir = os.path.join(task_dir, pairing, "tactile")
        if not os.path.isdir(tac_dir):
            print(f"[skip] no tactile dir for {pairing}")
            continue
        eps = sorted(
            d for d in os.listdir(tac_dir) if os.path.isdir(os.path.join(tac_dir, d))
        )
        if not eps:
            print(f"[skip] no episodes for {pairing}")
            continue
        chosen = random.sample(eps, min(args.n, len(eps)))
        print(f"\n=== {pairing}  ->  {chosen} ===")
        for ep in chosen:
            out_path = os.path.join(out_dir, f"{pairing}__{ep}.mp4")
            viz_episode(os.path.join(tac_dir, ep), out_path, fps=args.fps)

    print(f"\nDone. Videos in: {out_dir}")


if __name__ == "__main__":
    main()
