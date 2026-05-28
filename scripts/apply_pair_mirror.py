"""
Apply per-pair x-axis mirror to latent_data_processed_raw → latent_data_processed_mirror.

Convention: x-axis = the LONGER spatial axis (W = last dim). "Mirror about
the x-axis" means a reflection ACROSS that axis, which flips the H direction
(perpendicular to x) — i.e. torch.flip(x, dims=[-2]). Applies the same way
to daimon (3, 240, 320)  → flip 240-axis, and flexi (12, 16) → flip 12-axis.
eFlesh (15-vector) has no 2D spatial structure and is never mirrored.

Per-pair mirror policy (flexi always mirrored when present; daimon only in
de_pair where it must, since eflesh has no spatial axis to mirror):
  de_pair : mirror daimon. eflesh copied.
  ef_pair : mirror flexi.  eflesh copied.
  fd_pair : mirror flexi.  daimon copied.

Output is byte-identical to input for non-mirrored sensors (shutil.copy2).
For mirrored sensors a torch.flip is applied and the tensor is re-saved.

Usage:
  python scripts/apply_pair_mirror.py
  python scripts/apply_pair_mirror.py --pairs fd_pair --objects circle
"""

import argparse
import glob
import os
import re
import shutil

import torch
from tqdm import tqdm


PAIR_SENSORS = {
    "de_pair": ("daimon", "eflesh"),
    "ef_pair": ("eflesh", "flexi"),
    "fd_pair": ("daimon", "flexi"),
}

_FR_RE = re.compile(r"^ep(\d+)_fr(\d+)\.pt$")


def list_episodes(sensor_dir):
    eps = set()
    for f in os.listdir(sensor_dir):
        m = _FR_RE.match(f)
        if m:
            eps.add(int(m.group(1)))
    return sorted(eps)


def should_mirror(pair, sensor):
    if pair == "de_pair":
        return sensor == "daimon"
    if pair in ("ef_pair", "fd_pair"):
        return sensor == "flexi"
    raise ValueError(pair)


def mirror_tensor(x):
    """Mirror ACROSS the x-axis = flip the H direction (second-to-last dim).
    Works for (3, H, W), (H, W), etc."""
    return torch.flip(x, dims=[-2])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", default="/home/junsung/cross_modal_tactile/"
                   "tactile_benchmark/data/latent_data_processed_raw")
    p.add_argument("--out_root", default="/home/junsung/cross_modal_tactile/"
                   "tactile_benchmark/data/latent_data_processed_mirror")
    p.add_argument("--pairs", nargs="+", choices=list(PAIR_SENSORS),
                   default=list(PAIR_SENSORS))
    p.add_argument("--objects", nargs="+", default=None)
    p.add_argument("--clean", action="store_true",
                   help="Remove output dirs of selected pairs/objects before run.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    print(f"in_root  = {args.in_root}")
    print(f"out_root = {args.out_root}")
    print()

    mirrored = copied = 0

    for pair in args.pairs:
        sa, sb = PAIR_SENSORS[pair]
        in_pair_a = os.path.join(args.in_root, pair, sa)
        if not os.path.isdir(in_pair_a):
            print(f"[skip] missing {in_pair_a}")
            continue
        obj_dirs = sorted(o for o in os.listdir(in_pair_a)
                          if os.path.isdir(os.path.join(in_pair_a, o)))
        if args.objects:
            obj_dirs = [o for o in obj_dirs if o in args.objects]

        mirror_target = sa if should_mirror(pair, sa) else sb
        print(f"[{pair}] mirror {mirror_target} (H-axis flip), copy the other")

        for obj in obj_dirs:
            for sensor in (sa, sb):
                in_dir = os.path.join(args.in_root, pair, sensor, obj)
                out_dir = os.path.join(args.out_root, pair, sensor, obj)
                if args.clean and os.path.isdir(out_dir):
                    shutil.rmtree(out_dir)
                os.makedirs(out_dir, exist_ok=True)

                files = sorted(glob.glob(os.path.join(in_dir, "ep*_fr*.pt")))
                for f in tqdm(files, desc=f"  {pair}/{sensor}/{obj}", leave=False):
                    bn = os.path.basename(f)
                    dst = os.path.join(out_dir, bn)
                    if should_mirror(pair, sensor):
                        x = torch.load(f, map_location="cpu", weights_only=False)
                        torch.save(mirror_tensor(x), dst)
                        mirrored += 1
                    else:
                        shutil.copy2(f, dst)
                        copied += 1

    print()
    print(f"Done. mirrored {mirrored} files, copied {copied} files")
    print(f"Output -> {args.out_root}")


if __name__ == "__main__":
    main()
