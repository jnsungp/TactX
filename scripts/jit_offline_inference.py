#!/usr/bin/env python3
"""
Offline inference using a TorchScript-traced ACT policy (`policy_jit.pt`).

For each timestep in a held-out HDF5 episode:
  1. Reads (cam0, cam1) RGB images, qpos, ground-truth action.
  2. (Optional) Reads tactile (image / array / latent) according to the model's
     sensor config.
  3. Runs the JIT policy and denormalizes its first action prediction using
     `dataset_stats.pkl` saved at training time.
  4. Compares the first predicted action vs. the GT action and reports MAE/MSE
     per dimension. Optionally dumps a PNG with per-dim curves.

Example
-------
python scripts/jit_offline_inference.py \
    --jit_path  tactile_benchmark/checkpoints/pnp_daimon_new_ckpt/policy_jit.pt \
    --stats_path tactile_benchmark/checkpoints/pnp_daimon_new_ckpt/dataset_stats.pkl \
    --model_cfg tactile_benchmark/tactile_policy/configs/models/act_daimon.yaml \
    --hdf_path  tactile_benchmark/data/recordings/processed/pnp_daimon_val \
    --out_dir   tactile_benchmark/checkpoints/pnp_daimon_new_ckpt/eval_outputs
"""

import argparse
import glob
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# small utilities                                                             #
# --------------------------------------------------------------------------- #
def _to_chw_rgb(img):
    """uint8 image -> (3, H, W) uint8 numpy. Accepts JPEG bytes, HWC or CHW."""
    if img.ndim == 1:
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.transpose(2, 0, 1)
    if img.ndim == 3 and img.shape[-1] == 3:  # HWC
        return img.transpose(2, 0, 1)
    return img  # already CHW


def _unit_normalize(image_nchw_uint8):
    """uint8 NCHW (numpy) -> float32 NCHW in [0, 1] (torch)."""
    return torch.from_numpy(image_nchw_uint8).float() / 255.0


def _load_sensor_cfg(model_cfg_path):
    model_cfg_path = Path(model_cfg_path)
    with open(model_cfg_path, "r") as fp:
        cfg = yaml.safe_load(fp)

    sensor_cfg = None
    sensor_name = cfg.get("common", {}).get("sensor_config")
    if sensor_name:
        sensor_path = (
            model_cfg_path.parent.parent / "sensors" / f"{sensor_name}.yaml"
        )
        with open(sensor_path, "r") as fp:
            sensor_cfg = yaml.safe_load(fp)
    return cfg, sensor_cfg


def _resolve_paths(p):
    p = str(p)
    if Path(p).is_dir():
        return sorted(Path(p).glob("*.hdf5"))
    if any(c in p for c in "*?["):
        return sorted(Path(x) for x in glob.glob(p))
    return [Path(p)]


# --------------------------------------------------------------------------- #
# core inference                                                              #
# --------------------------------------------------------------------------- #
def run_episode(
    jit_policy,
    hdf_path,
    stats,
    camera_names,
    tactile_names,
    tactile_modality,
    device,
):
    with h5py.File(hdf_path, "r") as f:
        T = f["observation.state"].shape[0]
        emb = f.attrs.get("embodiment", "franka")

        preds, gts = [], []

        for t in range(T):
            # --- cameras --- #
            cams = []
            for name in camera_names:
                cams.append(_to_chw_rgb(f[f"observation.image.{name}"][t]))
            image_nchw = np.stack(cams, axis=0)[None]  # (1, N, 3, H, W)
            image_t = _unit_normalize(image_nchw).to(device)

            # --- qpos (normalized using training stats) --- #
            qpos = f["observation.state"][t]
            qpos_n = (qpos - stats["qpos_mean"]) / (stats["qpos_std"] + 1e-6)
            qpos_t = torch.from_numpy(qpos_n[None]).float().to(device)

            # --- tactile --- #
            extra = ()
            if tactile_modality == "image" and tactile_names:
                tac_imgs = []
                for name in tactile_names:
                    tac_imgs.append(_to_chw_rgb(f[f"observation.image.{name}"][t]))
                tac_nchw = np.stack(tac_imgs, axis=0)[None]
                tac_t = _unit_normalize(tac_nchw).to(device)
                extra = (tac_t,)
            elif tactile_modality in ("array", "state") and tactile_names:
                tac_list = []
                for name in tactile_names:
                    arr = np.atleast_1d(f[f"observation.{name}"][t]).astype(np.float32)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    tac_list.append(arr)
                tac = np.stack(tac_list, axis=0)  # (num_tactile, dim)
                if stats.get("tactile_mean") is not None:
                    tac = (tac - stats["tactile_mean"]) / (stats["tactile_std"] + 1e-6)
                tac_t = torch.from_numpy(tac[None]).float().to(device)
                extra = (tac_t,)

            # --- forward --- #
            with torch.no_grad():
                out = jit_policy(image_t, qpos_t, *extra)
            # JIT outputs (1, chunk, action_dim)
            pred_chunk = out[0].detach().cpu().numpy()
            pred = pred_chunk[0] * stats["action_std"] + stats["action_mean"]

            preds.append(pred)
            gts.append(f["action"][t])

    preds = np.asarray(preds)
    gts = np.asarray(gts)
    return preds, gts, emb


def report(preds, gts, real_dims=8):
    diff = preds - gts
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    real = slice(0, min(real_dims, preds.shape[1]))
    mae_r = float(np.mean(np.abs(diff[:, real])))
    mse_r = float(np.mean(diff[:, real] ** 2))

    print(f"  MAE (all dims):              {mae:.6f}")
    print(f"  MSE (all dims):              {mse:.6f}")
    print(f"  MAE (first {real_dims} dims):       {mae_r:.6f}")
    print(f"  MSE (first {real_dims} dims):       {mse_r:.6f}")

    print(f"  Per-dim MAE (first {real_dims} dims):")
    for i in range(real.stop):
        per = float(np.mean(np.abs(diff[:, i])))
        print(f"    dim {i}: {per:.6f}")
    return {"mae": mae, "mse": mse, "mae_real": mae_r, "mse_real": mse_r}


def maybe_plot(preds, gts, out_path, real_dims=8):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping figure.")
        return
    real = min(real_dims, preds.shape[1])
    fig, axes = plt.subplots(real, 1, figsize=(10, 2.0 * real), sharex=True)
    if real == 1:
        axes = [axes]
    for i in range(real):
        axes[i].plot(gts[:, i], label="GT", linewidth=1.5)
        axes[i].plot(preds[:, i], label="Pred", linewidth=1.0, alpha=0.8)
        axes[i].set_ylabel(f"dim {i}")
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("timestep")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved plot -> {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jit_path", required=True, help="path to policy_jit.pt")
    ap.add_argument("--stats_path", required=True, help="path to dataset_stats.pkl")
    ap.add_argument("--model_cfg", required=True, help="model yaml used at training")
    ap.add_argument("--hdf_path", required=True,
                    help="single HDF5, directory, or glob")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None,
                    help="if set, dumps per-episode npz + plot here")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--real_action_dims", type=int, default=8,
                    help="number of leading dims that are the actual action "
                         "(rest is padding to 128)")
    args = ap.parse_args()

    cfg, sensor_cfg = _load_sensor_cfg(args.model_cfg)
    camera_names = cfg["common"]["camera_names"]
    tactile_names = cfg["common"].get("tactile_names", []) or []
    tactile_modality = sensor_cfg["modality"] if sensor_cfg else "none"

    print(f"Loading JIT policy: {args.jit_path}")
    policy = torch.jit.load(args.jit_path, map_location=args.device).eval()

    print(f"Loading dataset stats: {args.stats_path}")
    with open(args.stats_path, "rb") as f:
        norm_stats = pickle.load(f)
    print(f"  embodiments: {list(norm_stats.keys())}")

    paths = _resolve_paths(args.hdf_path)
    if not paths:
        raise FileNotFoundError(args.hdf_path)

    out_dir = Path(args.out_dir) if args.out_dir else None

    all_metrics = []
    for i, p in enumerate(paths):
        print(f"\n[{i + 1}/{len(paths)}] {p}")
        with h5py.File(p, "r") as _f:
            emb_attr = _f.attrs.get("embodiment", next(iter(norm_stats)))
        stats = norm_stats[emb_attr] if emb_attr in norm_stats else norm_stats[next(iter(norm_stats))]

        preds, gts, emb = run_episode(
            policy,
            p,
            stats,
            camera_names,
            tactile_names,
            tactile_modality,
            args.device,
        )
        m = report(preds, gts, real_dims=args.real_action_dims)
        all_metrics.append(m)

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir / f"{p.stem}.npz", preds=preds, gts=gts, embodiment=emb)
            if not args.no_plot:
                maybe_plot(preds, gts, out_dir / f"{p.stem}.png",
                           real_dims=args.real_action_dims)

    if len(all_metrics) > 1:
        print("\n=== average over {} episodes ===".format(len(all_metrics)))
        for k in all_metrics[0]:
            print(f"  {k}: {np.mean([m[k] for m in all_metrics]):.6f}")


if __name__ == "__main__":
    main()
