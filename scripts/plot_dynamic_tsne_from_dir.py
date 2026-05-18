"""
Standalone dynamic t-SNE from an arbitrary dynamic_data directory.

Given a directory with the layout
    data_root/{daimon,eflesh,flexi}/ep****_fr****.pt

this script:
  1. Loads raw tactile .pt files for each modality.
  2. Applies the same z-score + resize normalisation used during training.
     Stats are read from <checkpoint_dir>/dataset_details.yaml by default
     (the exact stats the model was trained with).
  3. Encodes each frame through the trained cross_modal_vae3 encoder to get μ.
  4. Plots t-SNE of the three μ clouds and saves the PNG.

Usage:
    python scripts/plot_dynamic_tsne_from_dir.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_static_run_<name>_<TS> \
        --data_root tactile_benchmark/data/latent_space_teleop/pnp/dynamic_data \
        --out_path results/tsne_teleop_pnp.png

Optional:
    --stats_yaml         path/to/config.yaml   # override norm stats source
    --stats_dataset_name entry_name_in_yaml    # required when --stats_yaml is a multi-entry config
    --max_tsne_points    2000
    --batch_size         256
    --device             cuda:0
    --seed               0
    --daimon_deform_only                       # encode Daimon as ch0 replicated 3×
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.utils.models_setup import get_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENSOR_KEY_TO_FOLDER = {'D': 'daimon', 'E': 'eflesh', 'F': 'flexi'}
_SENSOR_COLOR = {'D': 'tab:blue', 'E': 'tab:orange', 'F': 'tab:green'}
_SENSOR_LABEL = {'D': 'Daimon',   'E': 'eFlesh',     'F': 'FlexiTac'}

_EP_FR_RE = re.compile(r'ep(\d+)_fr(\d+)\.pt$')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_ep_fr(name):
    m = _EP_FR_RE.search(os.path.basename(name))
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


class _RawSensorDataset(Dataset):
    """Loads raw .pt tactile frames from a single sensor directory.

    Applies z-score normalisation (and optional resize for Daimon).
    Returns (tensor, (ep_id, fr_id)).
    """

    def __init__(self, sensor_root, sensor_name, transform=None,
                 mean=None, std=None):
        files = sorted(
            glob.glob(os.path.join(sensor_root, '*.pt')),
            key=lambda p: _parse_ep_fr(p),
        )
        if not files:
            raise FileNotFoundError(f"No .pt files found under {sensor_root}")
        self.files       = files
        self.sensor_name = sensor_name
        self.transform   = transform
        self.mean = torch.tensor(mean, dtype=torch.float32) if mean is not None else None
        self.std  = torch.tensor(std,  dtype=torch.float32) if std  is not None else None

    def __len__(self):
        return len(self.files)

    def _zscore(self, x):
        if self.mean is None:
            return x
        m, s = self.mean.to(x.device), self.std.to(x.device)
        if x.dim() >= 2 and m.dim() == 1 and m.shape[0] == x.shape[0]:
            m = m.view(-1, *([1] * (x.dim() - 1)))
            s = s.view(-1, *([1] * (x.dim() - 1)))
        return (x - m) / (s + 1e-8)

    def __getitem__(self, idx):
        raw = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        if isinstance(raw, dict):
            if   'image'    in raw: raw = raw['image']
            elif 'magnetic' in raw: raw = raw['magnetic']
            elif 'flexi'    in raw: raw = raw['flexi']
            elif 'flexitac' in raw: raw = raw['flexitac']
            else:
                raise ValueError(
                    f"Unknown dict keys {list(raw.keys())} in {self.files[idx]}")
        x = raw.float()
        x = self._zscore(x)
        if self.transform is not None:
            x = self.transform(x)
        ep, fr = _parse_ep_fr(self.files[idx])
        return x, (ep, fr)


def _build_sensor_dataset(sensor_key, sensor_root, dataset_details, image_size):
    folder = _SENSOR_KEY_TO_FOLDER[sensor_key]
    transform = mean = std = None

    if folder == 'daimon':
        names = dataset_details.get('daimon_transforms') or []
        if 'resize' in names:
            transform = transforms.Resize((image_size, image_size), antialias=True)
        if 'normalize' in names:
            stats = dataset_details.get('daimon_stats')
            if stats:
                mean, std = stats['mean'], stats['std']

    elif folder == 'eflesh':
        names = dataset_details.get('eflesh_transforms') or []
        if 'normalize' in names:
            stats = dataset_details.get('eflesh_stats')
            if stats:
                mean, std = stats['mean'], stats['std']

    elif folder == 'flexi':
        names = dataset_details.get('flexitac_transforms') or []
        if 'normalize' in names:
            stats = dataset_details.get('flexitac_stats')
            if stats:
                m, s = stats['mean'], stats['std']
                mean = m if isinstance(m, (list, tuple)) else [float(m)]
                std  = s if isinstance(s, (list, tuple)) else [float(s)]

    return _RawSensorDataset(sensor_root, folder,
                             transform=transform, mean=mean, std=std)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_mu(dataset, sensor_key, model, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False)
    mus = []
    for x, _ in tqdm(loader, desc=f"encode_{sensor_key}"):
        out = model.forward_self(x.to(device), sensor_key)
        mus.append(out['mu'].cpu())
    return torch.cat(mus, dim=0)


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def _subsample(arr, n_max, seed=0):
    if len(arr) <= n_max:
        return arr
    rng = np.random.RandomState(seed)
    return arr[rng.choice(len(arr), n_max, replace=False)]


def _run_tsne(points, perplexity=30, seed=42):
    kw = dict(n_components=2, perplexity=perplexity, random_state=seed)
    ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
    kw['max_iter' if ver >= (1, 0) else 'n_iter'] = 1000
    return TSNE(**kw).fit_transform(points)


def plot_tsne_dynamic(dyn_mus, out_path, n_max=10000000):
    blocks, info = [], []
    for s in ('D', 'E', 'F'):
        arr = F.normalize(dyn_mus[s], dim=1).numpy()
        arr = _subsample(arr, n_max, seed=hash('dyn_' + s) & 0xFFFF)
        blocks.append(arr)
        info.append((s, len(arr)))

    combined = np.concatenate(blocks, axis=0)
    print(f"  t-SNE on {len(combined)} points "
          f"({', '.join(f'{s}:{n}' for s, n in info)})")
    coords = _run_tsne(combined)

    fig, ax = plt.subplots(figsize=(8, 7))
    off = 0
    for s, n in info:
        pts = coords[off:off + n]; off += n
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=_SENSOR_COLOR[s], alpha=0.5, s=14, label=_SENSOR_LABEL[s])

    ax.set_title('t-SNE dynamic μ')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Dynamic t-SNE from a dynamic_data directory.')
    parser.add_argument(
        '--checkpoint_dir', required=True,
        help='cross_modal_vae3 checkpoint directory (model.pt + model_details.yaml).')
    parser.add_argument(
        '--data_root',
        default='tactile_benchmark/data/latent_space_teleop/pnp/dynamic_data',
        help='Root with daimon/, eflesh/, flexi/ sub-dirs of ep****_fr****.pt files.')
    parser.add_argument(
        '--stats_yaml', default=None,
        help='Path to a datasets_config yaml to read normalisation stats from. '
             'Default: <checkpoint_dir>/dataset_details.yaml')
    parser.add_argument(
        '--stats_dataset_name', default=None,
        help='Entry name inside --stats_yaml (required for multi-entry configs '
             'like datasets_config_sachin.yaml).')
    parser.add_argument('--out_path', default=None,
                        help='Output PNG. Default: <data_root>/tsne_dynamic.png')
    parser.add_argument('--max_tsne_points', type=int, default=1000000000)
    parser.add_argument('--batch_size',      type=int, default=256)
    parser.add_argument('--device',          default='cuda:0')
    parser.add_argument('--seed',            type=int, default=0)
    parser.add_argument(
        '--daimon_deform_only', action='store_true',
        help='Encode Daimon as ch0 replicated 3× (deform-only checkpoint).')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = args.out_path or os.path.join(args.data_root, 'tsne_dynamic.png')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # ── Norm stats ────────────────────────────────────────────────────────────
    if args.stats_yaml:
        with open(args.stats_yaml) as f:
            raw_cfg = yaml.safe_load(f)
        if args.stats_dataset_name:
            if args.stats_dataset_name not in raw_cfg:
                raise KeyError(
                    f"'{args.stats_dataset_name}' not in {args.stats_yaml}. "
                    f"Available: {list(raw_cfg.keys())}")
            dataset_details = raw_cfg[args.stats_dataset_name]
            print(f"Norm stats : {args.stats_yaml}  [{args.stats_dataset_name}]")
        else:
            dataset_details = raw_cfg
            print(f"Norm stats : {args.stats_yaml}")
    else:
        stats_path = os.path.join(args.checkpoint_dir, 'dataset_details.yaml')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"{stats_path} not found. "
                f"Specify --stats_yaml / --stats_dataset_name explicitly.")
        with open(stats_path) as f:
            dataset_details = yaml.safe_load(f)
        print(f"Norm stats : {stats_path}  (checkpoint dataset_details)")

    image_size = dataset_details.get('image_size', 224)
    print(f"  image_size={image_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model('cross_modal_vae3', model_details).to(args.device)
    ckpt  = torch.load(os.path.join(args.checkpoint_dir, 'model.pt'),
                       map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model      : {args.checkpoint_dir}")

    if args.daimon_deform_only:
        _orig = model._enc
        def _enc_deform(x, sensor, _o=_orig):
            if sensor == 'D' and x.dim() == 4:
                x = x[:, 0:1].repeat(1, 3, 1, 1)
            return _o(x, sensor)
        model._enc = _enc_deform
        print("[daimon_deform_only] Daimon → ch0 ×3")

    # ── Encode each sensor ────────────────────────────────────────────────────
    dyn_mus = {}
    for sensor_key in ('D', 'E', 'F'):
        folder      = _SENSOR_KEY_TO_FOLDER[sensor_key]
        sensor_root = os.path.join(args.data_root, folder)
        if not os.path.isdir(sensor_root):
            raise FileNotFoundError(f"Not found: {sensor_root}")
        ds = _build_sensor_dataset(sensor_key, sensor_root,
                                   dataset_details, image_size)
        print(f"[{sensor_key}] {len(ds)} frames — encoding …")
        dyn_mus[sensor_key] = extract_mu(ds, sensor_key, model,
                                         args.device, args.batch_size)

    print("μ shapes:", {s: tuple(v.shape) for s, v in dyn_mus.items()})

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    plot_tsne_dynamic(dyn_mus, out_path, n_max=args.max_tsne_points)
    print("\nDone.")


if __name__ == '__main__':
    main()
