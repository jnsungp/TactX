"""
Cross-modal tactile preprocessing — RAW orientation, depth-only Daimon.

A simplified twin of scripts/preprocess_cross_modal_new.py with two changes:
  1. NO mirror/flip anywhere. Every sensor is kept in its raw orientation, so
     there is no per-pair / left-right ambiguity to track (in particular the
     fd_pair "which side was left" problem disappears).
  2. Daimon is reduced to its DEPTH channel only, replicated to 3 channels:
     (3, 240, 320) = [depth, depth, depth]. Shear_x / shear_y are dropped.

Everything else matches preprocess_cross_modal_new.py: it reads raw directories
named  left_{sensorA}_right_{sensorB}_{object}/  under --input_root and writes

OUTPUT:
  {output_root}/
    de_pair/ daimon/{object}/ep{NNNN}_fr{NNNN}.pt   eflesh/{object}/...
    ef_pair/ eflesh/{object}/...                     flexi/{object}/...
    fd_pair/ daimon/{object}/...                     flexi/{object}/...

Per-modality tensor shapes (single tensor per .pt file):
  daimon : (3, 240, 320)  float32   = [depth, depth, depth]   (raw orientation)
  eflesh : (15,)          float32   flattened from (5, 3)
  flexi  : (12, 16)       float32   raw 12x16 grid             (raw orientation)

Because the data is already depth-only and un-flipped, the training config for
this dataset should NOT apply the 'deform_only' transform and should NOT add any
per-pair 'rot180' / flip extra_transforms — those would double-correct.

Usage (run on the machine that holds the raw data):
  python scripts/preprocess_cross_modal_raw.py --clean
  python scripts/preprocess_cross_modal_raw.py --objects cylinder plug --pairs de_pair fd_pair ef_pair
  python scripts/preprocess_cross_modal_raw.py --dry_run

small_ball cylinder cylinder_vertical ellipse plane square
  

  for obj in small_ball cylinder cylinder_vertical ellipse plane square; do
  python scripts/preprocess_cross_modal_raw.py \
    --input_root  /home/junsung/cross_modal_tactile/tactile_benchmark/data/latent_data_new/$obj \
    --output_root /home/junsung/cross_modal_tactile/tactile_benchmark/data/latent_data_processed_raw \
    --objects $obj --pairs de_pair fd_pair ef_pair
  done

  rsync -avh --progress \
    /media/ExtHDD/sachin/latent_data_new_processed_raw/ \
    junsung@100.117.194.55:/home/junsung/cross_modal_tactile/tactile_benchmark/data/latent_data_new_processed_raw/
"""

import os
import re
import pickle
import shutil
import argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm


# ---- Constants -------------------------------------------------------------

DAIMON_H, DAIMON_W = 240, 320
DAIMON_N = DAIMON_H * DAIMON_W

SENSOR_ALIASES = {
    "daimon": "daimon",
    "eflesh": "eflesh",
    "flexi": "flexi",
}

SENSOR_PAIR_TO_NAME = {
    frozenset({"daimon", "eflesh"}): "de_pair",
    frozenset({"daimon", "flexi"}):  "fd_pair",
    frozenset({"eflesh", "flexi"}):  "ef_pair",
}

PAIR_SENSORS = {
    "de_pair": ("daimon", "eflesh"),
    "fd_pair": ("daimon", "flexi"),
    "ef_pair": ("eflesh", "flexi"),
}


# ---- Auto-discovery --------------------------------------------------------

_DIR_PATTERN = re.compile(
    r"^left_(?P<left>\w+?)_right_(?P<right>\w+?)_(?P<object>\w+)$"
)


def discover_sources(input_root):
    """Scan input_root and return a list of source dicts."""
    sources = []
    for name in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, name)
        if not os.path.isdir(path):
            continue
        m = _DIR_PATTERN.match(name)
        if not m:
            continue
        left  = SENSOR_ALIASES.get(m.group("left"))
        right = SENSOR_ALIASES.get(m.group("right"))
        obj   = m.group("object")
        if left is None or right is None:
            continue
        pair_name = SENSOR_PAIR_TO_NAME.get(frozenset({left, right}))
        if pair_name is None:
            continue
        sources.append({
            "dir": name,
            "pair": pair_name,
            "s0_sensor": left,
            "s1_sensor": right,
            "object": obj,
            "path": path,
        })
    return sources


# ---- Argument parsing ------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-modal preprocessing (raw orientation, depth-only daimon)")
    p.add_argument("--input_root", type=str,
                   default="/media/ExtHDD/sachin/latent_data_new/cylinder")
    p.add_argument("--output_root", type=str,
                   default="/media/ExtHDD/sachin/latent_data_new_processed_raw")
    p.add_argument("--dry_run", action="store_true",
                   help="Process only the first raw episode of each source")
    p.add_argument("--clean", action="store_true",
                   help="Remove output dirs of selected pairs/objects before run")
    p.add_argument("--pairs", nargs="+", default=None,
                   choices=list(PAIR_SENSORS.keys()),
                   help="Only process listed pairs (default: all)")
    p.add_argument("--objects", nargs="+", default=None,
                   help="Only process listed objects (default: all discovered)")
    return p.parse_args()


# ---- Sensor-aware tensor conversion (NO mirror) ----------------------------

def daimon_depth_chw(daimon_arr: np.ndarray) -> np.ndarray:
    """Raw-orientation Daimon: keep DEPTH (channel 0) only, replicate to 3 ch."""
    flat = daimon_arr.astype(np.float32).reshape(-1)
    depth = flat[0 * DAIMON_N:1 * DAIMON_N].reshape(DAIMON_H, DAIMON_W)
    chw = np.stack([depth, depth, depth], axis=0)   # (3, H, W)
    return np.ascontiguousarray(chw)


def flexi_to_2d(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.float32))


def eflesh_to_flat(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32).reshape(-1)


def frame_to_tensor(arr: np.ndarray, sensor: str) -> np.ndarray:
    if sensor == "daimon":
        return daimon_depth_chw(arr)
    if sensor == "eflesh":
        return eflesh_to_flat(arr)
    if sensor == "flexi":
        return flexi_to_2d(arr)
    raise ValueError(f"Unknown sensor: {sensor}")


# ---- Episode loaders -------------------------------------------------------

def load_perframe_episode(ep_dir: str):
    """Load all per-frame pickle files from an episode directory."""
    pkls = sorted([
        os.path.join(ep_dir, f) for f in os.listdir(ep_dir)
        if f.endswith(".pkl") and f != "metadata.pkl"
    ])
    s0_list, s1_list = [], []
    for p in pkls:
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        s0_list.append(d["s0"])
        s1_list.append(d["s1"])
    return s0_list, s1_list


def list_raw_episodes(in_dir: str):
    for name in sorted(os.listdir(in_dir)):
        if not name.startswith("episode_"):
            continue
        path = os.path.join(in_dir, name)
        if os.path.isdir(path):
            yield name, path


# ---- Per-source processing -------------------------------------------------

def process_source(src, args, ep_counter_key, ep_counters):
    """Process one raw directory. Returns stats dict."""
    pair_name = src["pair"]
    s0_sensor = src["s0_sensor"]
    s1_sensor = src["s1_sensor"]
    obj       = src["object"]
    in_dir    = src["path"]

    out_s0 = os.path.join(args.output_root, pair_name, s0_sensor, obj)
    out_s1 = os.path.join(args.output_root, pair_name, s1_sensor, obj)
    os.makedirs(out_s0, exist_ok=True)
    os.makedirs(out_s1, exist_ok=True)

    eps = list(list_raw_episodes(in_dir))
    if args.dry_run:
        eps = eps[:1]

    saved_eps = saved_frames = skipped = 0

    for ep_name, ep_path in tqdm(eps, desc=f"  {src['dir']}", leave=False):
        s0_seq, s1_seq = load_perframe_episode(ep_path)
        if len(s0_seq) == 0:
            skipped += 1
            continue

        ep_id = ep_counters[ep_counter_key]
        ep_counters[ep_counter_key] += 1

        for fr_idx, (raw_s0, raw_s1) in enumerate(zip(s0_seq, s1_seq)):
            t0 = frame_to_tensor(raw_s0, s0_sensor)
            t1 = frame_to_tensor(raw_s1, s1_sensor)
            fname = f"ep{ep_id:04d}_fr{fr_idx:04d}.pt"
            torch.save(torch.from_numpy(t0), os.path.join(out_s0, fname))
            torch.save(torch.from_numpy(t1), os.path.join(out_s1, fname))
            saved_frames += 1
        saved_eps += 1

    return {"saved_eps": saved_eps, "saved_frames": saved_frames, "skipped": skipped}


# ---- Entrypoint ------------------------------------------------------------

def main():
    args = parse_args()

    all_sources = discover_sources(args.input_root)
    if not all_sources:
        print(f"No valid source directories found under {args.input_root}")
        return

    if args.pairs:
        all_sources = [s for s in all_sources if s["pair"] in args.pairs]
    if args.objects:
        all_sources = [s for s in all_sources if s["object"] in args.objects]
    if not all_sources:
        print("No sources matched the selected --pairs / --objects filters.")
        return

    discovered_pairs   = sorted({s["pair"]   for s in all_sources})
    discovered_objects = sorted({s["object"] for s in all_sources})

    print("=" * 64)
    print("Cross-modal preprocessing — RAW orientation, depth-only daimon")
    print("=" * 64)
    print(f"  input_root    : {args.input_root}")
    print(f"  output_root   : {args.output_root}")
    print(f"  pairs         : {discovered_pairs}")
    print(f"  objects       : {discovered_objects}")
    print(f"  sources       : {len(all_sources)} directories")
    print(f"  mirror        : OFF (raw orientation)")
    print(f"  daimon        : depth-only, replicated to 3 channels")
    print(f"  dry_run       : {args.dry_run}")
    print()

    if args.clean:
        cleaned = set()
        for s in all_sources:
            pair_dir = os.path.join(args.output_root, s["pair"])
            for sensor in PAIR_SENSORS[s["pair"]]:
                obj_dir = os.path.join(pair_dir, sensor, s["object"])
                if os.path.isdir(obj_dir) and obj_dir not in cleaned:
                    print(f"  cleaning {obj_dir}")
                    shutil.rmtree(obj_dir)
                    cleaned.add(obj_dir)
    os.makedirs(args.output_root, exist_ok=True)

    ep_counters = defaultdict(int)
    summary = {}
    for src in all_sources:
        counter_key = (src["pair"], src["object"])
        print(f"[{src['pair']}/{src['object']}] {src['dir']}")
        stats = process_source(src, args, counter_key, ep_counters)
        summary[src["dir"]] = stats
        print(f"   eps={stats['saved_eps']}  frames={stats['saved_frames']}  "
              f"skipped={stats['skipped']}")

    print()
    print("=" * 64)
    print("Summary (episodes per pair/object):")
    for pair in discovered_pairs:
        for obj in discovered_objects:
            key = (pair, obj)
            if key in ep_counters:
                print(f"  {pair:8s} / {obj:10s} → {ep_counters[key]} episodes")
    print()
    print("Per-source breakdown:")
    for k, v in summary.items():
        print(f"  {k:50s}  eps={v['saved_eps']:3d}  "
              f"frames={v['saved_frames']:5d}  skipped={v['skipped']:3d}")
    print("=" * 64)


if __name__ == "__main__":
    main()
