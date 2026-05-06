from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from uniforce.config import UniForceConfig
from uniforce.dataset import CrossModalDataset, DEFAULT_OBJECTS
from uniforce.model import UniForce


def parse_args():
    p = argparse.ArgumentParser(description="Print UniForce val recon losses only")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--data_dir",
        type=str,
        default="/home/junsung/cross_modal_tactile/processed_data/cross_modal",
    )
    p.add_argument("--objects", type=str, nargs="+", default=list(DEFAULT_OBJECTS))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def eval_recon_only(model, loader, sL, sR, device):
    sums = {"n": 0, "LL": 0.0, "RR": 0.0, "LR": 0.0, "RL": 0.0}

    for batch in loader:
        cur_L = batch["cur_L"].to(device, non_blocking=True)
        cur_R = batch["cur_R"].to(device, non_blocking=True)

        out = model(cur_L, cur_R, sL, sR)
        B = cur_L.shape[0]

        sums["n"] += B
        sums["LL"] += B * F.l1_loss(out["recon_LL"], cur_L).item()
        sums["RR"] += B * F.l1_loss(out["recon_RR"], cur_R).item()
        sums["LR"] += B * F.l1_loss(out["recon_LR"], cur_R).item()
        sums["RL"] += B * F.l1_loss(out["recon_RL"], cur_L).item()

    n = max(sums["n"], 1)
    return {
        "n": sums["n"],
        "LL": sums["LL"] / n,
        "RR": sums["RR"] / n,
        "LR": sums["LR"] / n,
        "RL": sums["RL"] / n,
        "self_mean": (sums["LL"] + sums["RR"]) / (2 * n),
        "cross_mean": (sums["LR"] + sums["RL"]) / (2 * n),
        "overall_mean": (sums["LL"] + sums["RR"] + sums["LR"] + sums["RL"]) / (4 * n),
    }


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", UniForceConfig())

    model = UniForce(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = CrossModalDataset(
        args.data_dir,
        cfg,
        cfg.sensor_L,
        cfg.sensor_R,
        objects=tuple(args.objects),
        split="val",
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    m = eval_recon_only(model, loader, cfg.sensor_L, cfg.sensor_R, device)

    print("=" * 60)
    print("UniForce val reconstruction losses")
    print("=" * 60)
    print(f"checkpoint : {args.checkpoint}")
    print(f"sensor_L   : {cfg.sensor_L}")
    print(f"sensor_R   : {cfg.sensor_R}")
    print(f"objects    : {list(args.objects)}")
    print(f"n samples  : {m['n']}")
    print("-" * 60)
    print(f"LL  ({cfg.sensor_L}←{cfg.sensor_L}) : {m['LL']:.4f}")
    print(f"RR  ({cfg.sensor_R}←{cfg.sensor_R}) : {m['RR']:.4f}")
    print(f"LR  ({cfg.sensor_R}←{cfg.sensor_L}) : {m['LR']:.4f}")
    print(f"RL  ({cfg.sensor_L}←{cfg.sensor_R}) : {m['RL']:.4f}")
    print("-" * 60)
    print(f"self mean  : {m['self_mean']:.4f}")
    print(f"cross mean : {m['cross_mean']:.4f}")
    print(f"overall    : {m['overall_mean']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()