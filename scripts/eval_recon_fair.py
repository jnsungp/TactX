"""
Fair reconstruction-loss comparison between CrossModalVAE and UniForce.

The two models train with DIFFERENT normalizations:
  CrossModalVAE  — z-score (mean/std per dataset stats)
  UniForce       — min-max  [0, 1]

If you just print each model's training-space L1, numbers are NOT comparable:
  VAE  image L1 ≈ 0.10  (in z-score space, range roughly [-0.4, 2.7])
  UF   image L1 ≈ 0.04  (in [0,1] space)
→ VAE looks 2.5× worse even if reconstruction quality is identical.

Solution
--------
1. Load the same raw frames (no normalization).
2. Apply each model's own normalization, run forward, denormalize predictions
   back to the canonical [0,1] min-max space.
3. Compute L1 vs. the [0,1]-normalised GT in that shared space.
4. Report side-by-side.

Canonical [0,1] definition (same as UniForce's normalization):
  image:    x_01  = clip(x_raw,  0, 1)          (raw daimon ≈ already [0,1])
  magnetic: x_01  = clip((x_raw + 2048) / 4096, 0, 1)   (raw eflesh ~[-2048,2048])

Denormalization:
  VAE image    : x_zscore → x_raw = x_zscore * std_img + mean_img   → x_01 = clip(x_raw, 0, 1)
  VAE magnetic : x_zscore → x_raw = x_zscore * std_mag + mean_mag   → x_01 = clip((x_raw+2048)/4096, 0,1)
  UniForce     : output already in [0,1]  → no conversion needed

Usage:
  python scripts/eval_recon_fair.py \\
      --vae_checkpoint_dir checkpoints/cross_modal/cross_modal_vae/<run> \\
      --uniforce_checkpoint /home/junsung/cross_sensor_policy/checkpoints/uniforce_daimon_eflesh/uniforce_final.pt \\
      --device cuda:0

  # only one model:
  python scripts/eval_recon_fair.py --vae_checkpoint_dir ... --device cuda:0
  python scripts/eval_recon_fair.py --uniforce_checkpoint ... --device cuda:0

  python scripts/eval_recon_fair.py     --vae_checkpoint_dir checkpoints/cross_modal/cross_modal_vae/dataset_cross_modal_run_20260420_223654     --uniforce_checkpoint /home/junsung/cross_sensor_policy/checkpoints/uniforce_daimon_eflesh/uniforce_best.pt     --device cuda:0

  
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Shared constants ──────────────────────────────────────────────────────────
_EP_FR_RE = re.compile(r"(ep\d+)_fr(\d+)\.pt$")
DATA_DIR = "/home/junsung/cross_modal_tactile/processed_data/cross_modal"
DATASET_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "datasets_config.yaml",
)
VAL_OBJECTS = ["can_val", "cylinder_val", "large_ball_val", "pen_val", "small_ball_val"]

# Magnetic canonical range (UniForce convention): [-2048, 2048] → [0, 1]
MAG_MIN, MAG_MAX = -2048.0, 2048.0


# ══════════════════════════════════════════════════════════════════════════════
# Raw-frame loader (no normalization)
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_frame(daimon_fp: str, eflesh_fp: str):
    """Load raw tensors without any normalization.

    Returns:
        img_raw : (1, H, W)  float, values ≈ [0, 1]
        mag_raw : (15,)      float, values ≈ [-2048, 2048]
    """
    d = torch.load(daimon_fp, weights_only=False)
    e = torch.load(eflesh_fp, weights_only=False)
    img_raw = d["image"].float()
    if img_raw.dim() == 2:
        img_raw = img_raw.unsqueeze(0)
    mag_raw = e["magnetic"].float()
    return img_raw, mag_raw


def raw_to_canonical(img_raw: torch.Tensor, mag_raw: torch.Tensor):
    """Convert raw tensors to the shared canonical [0,1] space.

    canonical image   : clip(img_raw, 0, 1)
    canonical magnetic: clip((mag + 2048) / 4096, 0, 1)
    """
    img_01 = img_raw.clamp(0.0, 1.0)
    mag_01 = ((mag_raw - MAG_MIN) / (MAG_MAX - MAG_MIN + 1e-9)).clamp(0.0, 1.0)
    return img_01, mag_01


# ══════════════════════════════════════════════════════════════════════════════
# CrossModalVAE helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_vae(checkpoint_dir: str, device: torch.device):
    from joint_embedding_learning.utils.models_setup import get_model
    with open(os.path.join(checkpoint_dir, "model_details.yaml")) as f:
        model_details = yaml.safe_load(f)
    model = get_model("cross_modal_vae", model_details)
    state = torch.load(os.path.join(checkpoint_dir, "model.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, model_details


def load_vae_stats(dataset_config: str = DATASET_CONFIG, dataset_name: str = "dataset_cross_modal"):
    from joint_embedding_learning.utils.utils import get_config_details
    cfg = get_config_details(dataset_config, dataset_name)
    img_mean = torch.tensor(cfg["daimon_stats"]["mean"])  # (3,)
    img_std  = torch.tensor(cfg["daimon_stats"]["std"])
    mag_mean = torch.tensor(cfg["eflesh_stats"]["mean"])  # (15,)
    mag_std  = torch.tensor(cfg["eflesh_stats"]["std"])
    img_size = cfg.get("image_size", 224)
    return img_mean, img_std, mag_mean, mag_std, img_size


def vae_preprocess(img_raw: torch.Tensor, mag_raw: torch.Tensor,
                   img_mean, img_std, mag_mean, mag_std, img_size: int):
    """Apply VAE's z-score normalization."""
    img = img_raw.clamp(0.0, 1.0)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(img_mean, img_std),
    ])
    img_z = tf(img)
    mag_z = (mag_raw - mag_mean) / (mag_std + 1e-8)
    return img_z, mag_z


def vae_denorm_to_canonical(pred_img_z: torch.Tensor, pred_mag_z: torch.Tensor,
                             img_mean, img_std, mag_mean, mag_std):
    """De-normalize VAE predictions from z-score → raw → canonical [0,1].

    pred_img_z : (3, H, W)  in z-score image space
    pred_mag_z : (15,)      in z-score magnetic space
    """
    m = img_mean.to(pred_img_z).view(-1, 1, 1)
    s = img_std.to(pred_img_z).view(-1, 1, 1)
    img_raw = pred_img_z * s + m           # z-score → raw  ≈ [0,1]
    img_01  = img_raw.clamp(0.0, 1.0)     # canonical

    mag_raw = pred_mag_z * mag_std.to(pred_mag_z) + mag_mean.to(pred_mag_z)
    mag_01  = ((mag_raw - MAG_MIN) / (MAG_MAX - MAG_MIN + 1e-9)).clamp(0.0, 1.0)
    return img_01, mag_01


@torch.no_grad()
def eval_vae_fair(model, objects: list[str],
                  img_mean, img_std, mag_mean, mag_std, img_size: int,
                  daimon_base: str, eflesh_base: str,
                  device: torch.device, batch_size: int = 32):
    """Run CrossModalVAE and compute L1 in canonical [0,1] space."""
    sums = {"img_self": 0.0, "img_cross": 0.0,
            "mag_self": 0.0, "mag_cross": 0.0, "n": 0}

    for obj in objects:
        d_dir = os.path.join(daimon_base, obj)
        e_dir = os.path.join(eflesh_base, obj)
        if not os.path.isdir(d_dir):
            continue
        frames = [(os.path.join(d_dir, n), os.path.join(e_dir, n))
                  for n in sorted(os.listdir(d_dir)) if _EP_FR_RE.match(n)]
        if not frames:
            continue

        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            xi_z_list, xj_z_list = [], []
            xi_01_list, xj_01_list = [], []
            for dfp, efp in batch:
                img_raw, mag_raw = load_raw_frame(dfp, efp)
                img_01, mag_01 = raw_to_canonical(img_raw, mag_raw)
                img_z, mag_z = vae_preprocess(img_raw, mag_raw,
                                              img_mean, img_std, mag_mean, mag_std, img_size)
                xi_z_list.append(img_z)
                xj_z_list.append(mag_z)
                # GT in canonical space (resize to match model output size)
                gt_img = F.interpolate(img_01.unsqueeze(0).expand(1, 3, -1, -1)
                                       .float(), size=(img_size, img_size),
                                       mode="bilinear", align_corners=False).squeeze(0)
                xi_01_list.append(gt_img)
                xj_01_list.append(mag_01)

            x_i_z  = torch.stack(xi_z_list).to(device)
            x_j_z  = torch.stack(xj_z_list).to(device)
            gt_i_01 = torch.stack(xi_01_list).to(device)   # GT canonical
            gt_j_01 = torch.stack(xj_01_list).to(device)

            out = model(x_i_z, x_j_z)

            # Denormalize VAE predictions → canonical [0,1]
            B = x_i_z.shape[0]
            for b in range(B):
                ri_01, _ = vae_denorm_to_canonical(
                    out["recon_ii"][b].cpu(), out["recon_jj"][b].cpu(),
                    img_mean, img_std, mag_mean, mag_std)
                rj_01_self, rj_01_cross = (
                    ((out["recon_jj"][b].cpu() * mag_std + mag_mean - MAG_MIN)
                     / (MAG_MAX - MAG_MIN + 1e-9)).clamp(0, 1),
                    ((out["recon_ij"][b].cpu() * mag_std + mag_mean - MAG_MIN)
                     / (MAG_MAX - MAG_MIN + 1e-9)).clamp(0, 1),
                )
                ri_cross_01, _ = vae_denorm_to_canonical(
                    out["recon_ji"][b].cpu(), out["recon_ij"][b].cpu(),
                    img_mean, img_std, mag_mean, mag_std)

                gt_i = gt_i_01[b].cpu()
                gt_j = gt_j_01[b].cpu()
                sums["img_self"]  += F.l1_loss(ri_01,       gt_i).item()
                sums["img_cross"] += F.l1_loss(ri_cross_01, gt_i).item()
                sums["mag_self"]  += F.l1_loss(rj_01_self,  gt_j).item()
                sums["mag_cross"] += F.l1_loss(rj_01_cross, gt_j).item()
                sums["n"] += 1

    n = max(sums["n"], 1)
    return {k: sums[k] / n for k in ("img_self", "img_cross", "mag_self", "mag_cross")}


# ══════════════════════════════════════════════════════════════════════════════
# UniForce helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_uniforce(checkpoint_path: str, device: torch.device):
    sys.path.insert(0, "/home/junsung/cross_sensor_policy")
    from uniforce.config import UniForceConfig
    from uniforce.model import UniForce
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", UniForceConfig())
    model = UniForce(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def uniforce_preprocess_batch(img_raws, mag_raws, img_size: int = 64):
    """Min-max normalize to [0,1] and resize, matching UniForce's _load_and_normalize."""
    xi_list, xj_list = [], []
    for img_raw, mag_raw in zip(img_raws, mag_raws):
        img = img_raw.clamp(0.0, 1.0)
        if tuple(img.shape[-2:]) != (img_size, img_size):
            img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size),
                                mode="bilinear", align_corners=False).squeeze(0)
        xi_list.append(img)
        mag_01 = ((mag_raw - MAG_MIN) / (MAG_MAX - MAG_MIN + 1e-9)).clamp(0.0, 1.0)
        xj_list.append(mag_01)
    return torch.stack(xi_list), torch.stack(xj_list)


@torch.no_grad()
def eval_uniforce_fair(model, cfg, objects: list[str],
                       daimon_base: str, eflesh_base: str,
                       device: torch.device, batch_size: int = 32):
    """Run UniForce and compute L1 in canonical [0,1] space.

    UniForce output is ALREADY in [0,1] — no denormalization needed.
    Just evaluate using the same [0,1] GT.
    """
    img_size = cfg.img_height   # 64
    sL, sR = cfg.sensor_L, cfg.sensor_R

    sums = {"img_self": 0.0, "img_cross": 0.0,
            "mag_self": 0.0, "mag_cross": 0.0, "n": 0}

    for obj in objects:
        # UniForce val split uses obj names without _val suffix in DEFAULT_OBJECTS,
        # but we pass the same val directory names as the VAE comparison.
        d_dir = os.path.join(daimon_base, obj)
        e_dir = os.path.join(eflesh_base, obj)
        if not os.path.isdir(d_dir):
            continue
        frames = [(os.path.join(d_dir, n), os.path.join(e_dir, n))
                  for n in sorted(os.listdir(d_dir)) if _EP_FR_RE.match(n)]
        if not frames:
            continue

        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            img_raws, mag_raws = [], []
            for dfp, efp in batch:
                ir, mr = load_raw_frame(dfp, efp)
                img_raws.append(ir)
                mag_raws.append(mr)

            x_i, x_j = uniforce_preprocess_batch(img_raws, mag_raws, img_size)
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            out = model(x_i, x_j, sL, sR)

            # Output already in [0,1] — GT is the same normalized input.
            sums["img_self"]  += F.l1_loss(out["recon_LL"], x_i).item() * len(batch)
            sums["img_cross"] += F.l1_loss(out["recon_RL"], x_i).item() * len(batch)
            sums["mag_self"]  += F.l1_loss(out["recon_RR"], x_j).item() * len(batch)
            sums["mag_cross"] += F.l1_loss(out["recon_LR"], x_j).item() * len(batch)
            sums["n"] += len(batch)

    n = max(sums["n"], 1)
    return {k: sums[k] / n for k in ("img_self", "img_cross", "mag_self", "mag_cross")}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fair recon-loss comparison: CrossModalVAE vs UniForce (canonical [0,1] L1)")
    p.add_argument("--vae_checkpoint_dir", type=str, default=None,
                   help="CrossModalVAE run directory (model.pt + model_details.yaml)")
    p.add_argument("--uniforce_checkpoint", type=str, default=None,
                   help="UniForce checkpoint .pt file")
    p.add_argument("--objects", type=str, nargs="+", default=VAL_OBJECTS)
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def _print_result(name: str, m: dict, img_size: str):
    print(f"\n{'─'*50}")
    print(f"  {name}  (GT & pred in canonical [0,1] space,  img @ {img_size})")
    print(f"{'─'*50}")
    print(f"  image  self  (D_img(z_i) vs x_i): {m['img_self']:.4f}")
    print(f"  image  cross (D_img(z_j) vs x_i): {m['img_cross']:.4f}")
    print(f"  mag    self  (D_mag(z_j) vs x_j): {m['mag_self']:.4f}")
    print(f"  mag    cross (D_mag(z_i) vs x_j): {m['mag_cross']:.4f}")
    img_gap = m["img_cross"] - m["img_self"]
    mag_gap = m["mag_cross"] - m["mag_self"]
    print(f"  image  gap   (cross - self):        {img_gap:+.4f}"
          f"  {'↑ cross harder (expected)' if img_gap > 0 else '↓ cross easier (unusual)'}")
    print(f"  mag    gap   (cross - self):        {mag_gap:+.4f}"
          f"  {'↑ cross harder (expected)' if mag_gap > 0 else '↓ cross easier (unusual)'}")


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    daimon_base = os.path.join(args.data_dir, "daimon")
    eflesh_base = os.path.join(args.data_dir, "eflesh")

    print("=" * 60)
    print("Fair Reconstruction Loss Comparison")
    print("Canonical space: [0,1] min-max for both image and magnetic")
    print(f"  image   : raw ≈ [0,1]  → canonical = clip(raw, 0, 1)")
    print(f"  magnetic: raw ≈ [-2048,2048] → canonical = (raw+2048)/4096")
    print(f"Objects: {args.objects}")
    print("=" * 60)

    results = {}

    # ── CrossModalVAE ──
    if args.vae_checkpoint_dir:
        ckpt_dir = args.vae_checkpoint_dir
        if not os.path.isabs(ckpt_dir):
            ckpt_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ckpt_dir)
        print(f"\n[CrossModalVAE] loading from {ckpt_dir}")
        vae, vae_details = load_vae(ckpt_dir, device)
        img_mean, img_std, mag_mean, mag_std, img_size = load_vae_stats()
        print(f"  projection_dim={vae_details['projection_dim']}, "
              f"resnet={vae_details['resnet']}, img_size={img_size}")
        m_vae = eval_vae_fair(
            vae, args.objects, img_mean, img_std, mag_mean, mag_std, img_size,
            daimon_base, eflesh_base, device, args.batch_size,
        )
        results["CrossModalVAE"] = (m_vae, f"{img_size}×{img_size}")

    # ── UniForce ──
    if args.uniforce_checkpoint:
        print(f"\n[UniForce] loading from {args.uniforce_checkpoint}")
        uf_model, uf_cfg = load_uniforce(args.uniforce_checkpoint, device)
        print(f"  img_size={uf_cfg.img_height}×{uf_cfg.img_width}")
        m_uf = eval_uniforce_fair(
            uf_model, uf_cfg, args.objects,
            daimon_base, eflesh_base, device, args.batch_size,
        )
        results["UniForce"] = (m_uf, f"{uf_cfg.img_height}×{uf_cfg.img_width}")

    # ── Print ──
    print("\n" + "=" * 60)
    print("RESULTS  (canonical [0,1] L1 — directly comparable)")
    for name, (m, img_sz) in results.items():
        _print_result(name, m, img_sz)

    if len(results) == 2:
        names = list(results.keys())
        ma, mb = results[names[0]][0], results[names[1]][0]
        print(f"\n{'─'*50}")
        print(f"  DELTA  ({names[0]} − {names[1]})")
        print(f"{'─'*50}")
        for k in ("img_self", "img_cross", "mag_self", "mag_cross"):
            diff = ma[k] - mb[k]
            winner = names[1] if diff > 0 else names[0]
            print(f"  {k:<13}: {diff:+.4f}  (lower = better → {winner})")

    print("\n" + "=" * 60)
    print("NOTE: image L1 is computed at each model's native resolution.")
    print("  VAE  (224×224): finer detail → potentially higher L1 than 64×64.")
    print("  UF   ( 64× 64): coarser → possibly lower L1 due to smoothing.")
    print("  Value range is now fair; resolution difference is acknowledged.")
    print("=" * 60)


if __name__ == "__main__":
    main()
