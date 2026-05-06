import os
import re
import torch
import yaml
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from joint_embedding_learning.utils.models_setup import get_model
from joint_embedding_learning.utils.utils import get_config_details

_EP_FR_RE = re.compile(r"(ep\d+)_fr(\d+)\.pt$")


# ─────────────────────────────────────────
# Load
# ─────────────────────────────────────────
def load_checkpoint(checkpoint_dir, device):
    with open(os.path.join(checkpoint_dir, "model_details.yaml")) as f:
        model_details = yaml.safe_load(f)

    model = get_model("cross_modal_vae", model_details)
    state = torch.load(os.path.join(checkpoint_dir, "model.pt"),
                       map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    return model, model_details


def build_transforms(dataset_details):
    img_size = dataset_details.get("image_size", 224)

    mean = dataset_details["daimon_stats"]["mean"]
    std = dataset_details["daimon_stats"]["std"]

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(torch.tensor(mean), torch.tensor(std)),
    ])

    e_mean = torch.tensor(dataset_details["eflesh_stats"]["mean"])
    e_std  = torch.tensor(dataset_details["eflesh_stats"]["std"])

    return tf, e_mean, e_std


def load_frame(dfp, efp, tf, e_mean, e_std):
    d = torch.load(dfp)
    e = torch.load(efp)

    img = d["image"].float()
    if img.dim() == 2:
        img = img.unsqueeze(0)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = tf(img)

    mag = e["magnetic"].float()
    mag = (mag - e_mean) / (e_std + 1e-8)

    return img, mag


# ─────────────────────────────────────────
# Eval
# ─────────────────────────────────────────
@torch.no_grad()
def eval_all(model, objects, daimon_base, eflesh_base,
             tf, e_mean, e_std, device, batch_size=64):

    sums = {"ii": 0, "ji": 0, "jj": 0, "ij": 0, "n": 0}

    for obj in objects:
        d_dir = os.path.join(daimon_base, obj)
        e_dir = os.path.join(eflesh_base, obj)

        frames = []
        for name in os.listdir(d_dir):
            if _EP_FR_RE.match(name):
                frames.append((
                    os.path.join(d_dir, name),
                    os.path.join(e_dir, name)
                ))

        if not frames:
            continue

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]

            x_i_list, x_j_list = [], []
            for dfp, efp in batch:
                xi, xj = load_frame(dfp, efp, tf, e_mean, e_std)
                x_i_list.append(xi)
                x_j_list.append(xj)

            x_i = torch.stack(x_i_list).to(device)
            x_j = torch.stack(x_j_list).to(device)

            out = model(x_i, x_j)
            B = x_i.size(0)

            sums["ii"] += B * F.l1_loss(out["recon_ii"], x_i).item()
            sums["ji"] += B * F.l1_loss(out["recon_ji"], x_i).item()
            sums["jj"] += B * F.l1_loss(out["recon_jj"], x_j).item()
            sums["ij"] += B * F.l1_loss(out["recon_ij"], x_j).item()
            sums["n"]  += B

    n = sums["n"]

    print("\n========== VAL RESULT ==========")
    print(f"Total frames: {n}")
    print(f"img-self  : {sums['ii']/n:.4f}")
    print(f"img-cross : {sums['ji']/n:.4f}")
    print(f"mag-self  : {sums['jj']/n:.4f}")
    print(f"mag-cross : {sums['ij']/n:.4f}")
    print("================================\n")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    checkpoint_dir = "/home/junsung/cross_sensor_policy/checkpoints/uniforce_daimon_eflesh"
    dataset_config = "config/datasets_config.yaml"
    dataset_name = "dataset_cross_modal"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = load_checkpoint(checkpoint_dir, device)
    dataset_details = get_config_details(dataset_config, dataset_name)

    tf, e_mean, e_std = build_transforms(dataset_details)

    daimon_base = dataset_details["daimon_path"]
    eflesh_base = dataset_details["eflesh_path"]

    objects = dataset_details["tools"]["val_tools"]  # 🔥 전체 val set

    eval_all(model, objects, daimon_base, eflesh_base,
             tf, e_mean, e_std, device)


if __name__ == "__main__":
    main()