"""
Sachin variant of the tri-modal paired dataset loader. Adds per-pair
geometric correction transforms (fliplr / flipud / rot180) applied at
__getitem__ time so wrong-axis mirroring baked into preprocessed .pt
files can be corrected without rewriting the data.

Reads dataset config from config/datasets_config_sachin.yaml. Reuses
CrossModalPairDataset from the upstream loader; does not edit any
upstream file.

Per-pair config schema (in datasets_config_sachin.yaml):
    pairs:
      <pair_key>:
        sensor_a: 'daimon' | 'eflesh' | 'flexi'
        sensor_b: 'daimon' | 'eflesh' | 'flexi'
        root: '<pair_root_dir>'
        extra_transforms_a: ['rot180']   # optional, list of flip/rot ops
        extra_transforms_b: ['rot180']   # optional, list of flip/rot ops

Supported flip/rot names: 'fliplr', 'flipud', 'rot180'.
"""

import os
from collections import defaultdict

import torch
import yaml
from torch.utils.data import random_split, Subset
from torchvision import transforms

from joint_embedding_learning.data.datasets import CrossModalPairDataset


_FLIPROT_NAMES = ('fliplr', 'flipud', 'rot180')


class _RuntimeFlipRot:
    """Deterministic flip/rotation transform usable inside transforms.Compose.

        fliplr → mirror about the vertical axis  (left/right swap)
        flipud → mirror about the horizontal axis (top/bottom swap)
        rot180 → both, equivalent to undo-fliplr + apply-flipud

    Works for (C, H, W), (H, W), and (D,) tensors via negative-axis indices.
    """

    def __init__(self, mode):
        if mode not in _FLIPROT_NAMES:
            raise ValueError(f"Unknown flip/rot mode '{mode}', expected one of {_FLIPROT_NAMES}")
        self.mode = mode

    def __call__(self, x):
        if x.dim() < 2:
            return torch.flip(x, dims=[-1]) if self.mode == 'fliplr' else x
        if self.mode == 'fliplr':
            return torch.flip(x, dims=[-1])
        if self.mode == 'flipud':
            return torch.flip(x, dims=[-2])
        return torch.flip(x, dims=[-2, -1])


class _DeformOnlyGray:
    """Daimon channel reduction: keep channel 0 (normal deformation), drop the
    two shear channels, and replicate ch0 to 3 identical channels.

    The Daimon .pt stores (3, H, W) = [deform, shear_x, shear_y]. This makes
    the encoder see [deform, deform, deform] — a grayscale-deform input — so
    the shared latent represents only normal deformation (what FlexiTac, the
    normal-only sensor, can also represent).

    Applied AFTER z-score (CrossModalPairDataset z-scores the real 3-channel
    image first, so ch0 is correctly normalized with ch0's stats before it is
    selected here — no daimon_stats change needed).
    """

    def __call__(self, x):
        if x.dim() == 2:                 # (H, W) -> (1, H, W)
            x = x.unsqueeze(0)
        return x[0:1].repeat(3, 1, 1)    # (3, H, W), all = channel 0


def _build_sensor_transforms_sachin(dataset_details, sensor_name, image_size, extra_names=None):
    """Construct (transforms, mean, std) for a sensor, optionally appending
    deterministic flip/rot transforms named in `extra_names`."""
    transforms_list = []
    mean = None
    std = None

    if sensor_name == 'daimon':
        names = dataset_details.get('daimon_transforms', None) or []
        if 'resize' in names:
            transforms_list.append(transforms.Resize((image_size, image_size), antialias=True))
        if 'normalize' in names:
            stats = dataset_details.get('daimon_stats', None)
            if stats is not None:
                mean = stats['mean']
                std = stats['std']
        # 'deform_only': keep ch0 (deform), drop shear, replicate to 3 ch.
        # Runs after resize; z-score already applied per-channel in _load_one.
        if 'deform_only' in names:
            transforms_list.append(_DeformOnlyGray())
    elif sensor_name == 'eflesh':
        names = dataset_details.get('eflesh_transforms', None) or []
        if 'normalize' in names:
            stats = dataset_details.get('eflesh_stats', None)
            if stats is not None:
                mean = stats['mean']
                std = stats['std']
    elif sensor_name == 'flexi':
        names = dataset_details.get('flexitac_transforms', None) or []
        if 'normalize' in names:
            stats = dataset_details.get('flexitac_stats', None)
            if stats is not None:
                m = stats['mean']
                s = stats['std']
                mean = m if isinstance(m, (list, tuple)) else [float(m)]
                std = s if isinstance(s, (list, tuple)) else [float(s)]
    else:
        raise ValueError(f"Unknown sensor: {sensor_name}")

    # Flip/rot at the tail of the chain. z-score is applied separately in
    # CrossModalPairDataset._apply_zscore (per-pixel), and commutes with
    # these geometric ops.
    for nm in (extra_names or ()):
        if nm not in _FLIPROT_NAMES:
            raise ValueError(
                f"extra_transforms entry '{nm}' must be one of {_FLIPROT_NAMES}"
            )
        transforms_list.append(_RuntimeFlipRot(nm))

    transform = transforms.Compose(transforms_list) if transforms_list else None
    return transform, mean, std


def get_cross_modal_tri_datasets_sachin(dataset_name, dataset_details,
                                        data_split='train', device='cpu',
                                        objects=None, pair_strides=None):
    """Per-pair-transform-aware tri-modal dataset builder.

    Mirrors the upstream get_cross_modal_tri_datasets but reads optional
    per-pair `extra_transforms_a` / `extra_transforms_b` from each pair's
    config.

    Returns:
        train_datasets : dict {'de': ds, 'ef': ds, 'fd': ds}
        val_datasets   : dict {'de': ds_val, 'ef': ds_val, 'fd': ds_val}
        empty_test     : {}
    """
    data_root = dataset_details['data_root']
    image_size = dataset_details.get('image_size', 224)
    pairs = dataset_details['pairs']
    train_frac = float(dataset_details.get('train_frac', 0.8))

    train_datasets = {}
    val_datasets = {}

    print(f"[datasets_sachin] building '{dataset_name}' with per-pair transforms:")
    for pair_key, pair_cfg in pairs.items():
        sensor_a = pair_cfg['sensor_a']
        sensor_b = pair_cfg['sensor_b']
        pair_root = os.path.join(data_root, pair_cfg['root'])
        extra_a = pair_cfg.get('extra_transforms_a', None)
        extra_b = pair_cfg.get('extra_transforms_b', None)

        ta, ma, sa = _build_sensor_transforms_sachin(
            dataset_details, sensor_a, image_size, extra_names=extra_a,
        )
        tb, mb, sb = _build_sensor_transforms_sachin(
            dataset_details, sensor_b, image_size, extra_names=extra_b,
        )

        print(f"  [{pair_key}] {sensor_a:6s} ↔ {sensor_b:6s} | "
              f"extra_a={extra_a or '[]'}  extra_b={extra_b or '[]'}")

        stride = int((pair_strides or {}).get(pair_key, 1))
        if stride > 1:
            print(f"  [{pair_key}] frame_stride={stride} "
                  f"(keep every {stride}th frame per episode)")
        ds_full = CrossModalPairDataset(
            pair_root, sensor_a, sensor_b,
            transforms_a=ta, transforms_b=tb,
            mean_a=ma, std_a=sa, mean_b=mb, std_b=sb,
            device=device, objects=objects, frame_stride=stride,
        )

        # Train/val split. Two modes, selected by the dataset_details key
        # 'episode_val_split' (default False = original behavior):
        #
        #   False -> frame-level random_split (seed 0). Original behavior.
        #            Caveat: adjacent episode frames are near-duplicates, so
        #            train/val leak near-twins.
        #   True  -> episode-level split: whole (object, episode) groups go to
        #            train OR val, never split. Deterministic (sorted +
        #            modular): per object, every val_stride-th episode -> val,
        #            so every object is represented in val and there is no
        #            near-duplicate leakage.
        #            If 'val_episode_ids' is set (list of int), those specific
        #            episode indices go to val regardless of stride logic.
        if not bool(dataset_details.get('episode_val_split', False)):
            n = len(ds_full)
            train_len = int(train_frac * n)
            val_len = n - train_len
            ds_train, ds_val = random_split(
                ds_full, [train_len, val_len],
                generator=torch.Generator().manual_seed(0),
            )
            train_datasets[pair_key] = ds_train
            val_datasets[pair_key] = ds_val
        else:
            groups = defaultdict(list)        # (obj_id, ep_id) -> sample idxs
            for i in range(len(ds_full)):
                groups[(ds_full.object_per_sample[i],
                        ds_full.episode_per_sample[i])].append(i)
            by_obj = defaultdict(list)        # obj_id -> [(obj,ep) keys]
            for key in groups:
                by_obj[key[0]].append(key)
            train_idx, val_idx = [], []

            explicit_val_eps = dataset_details.get('val_episode_ids', None)
            if explicit_val_eps is not None:
                val_ep_set = set(int(e) for e in explicit_val_eps)
                for key, idxs in groups.items():
                    if key[1] in val_ep_set:
                        val_idx.extend(idxs)
                    else:
                        train_idx.extend(idxs)
                print(f"  [{pair_key}] episode-level split (explicit): "
                      f"{len(train_idx)} train / {len(val_idx)} val frames "
                      f"over {len(groups)} episodes "
                      f"(val eps = {sorted(val_ep_set)})")
            else:
                val_stride = max(2, round(1.0 / max(1e-6, 1.0 - train_frac)))
                for obj_id in sorted(by_obj):
                    for j, key in enumerate(sorted(by_obj[obj_id])):
                        (val_idx if j % val_stride == 0 else train_idx).extend(
                            groups[key])
                print(f"  [{pair_key}] episode-level split: "
                      f"{len(train_idx)} train / {len(val_idx)} val frames "
                      f"over {len(groups)} episodes (val = every {val_stride}th)")

            train_idx.sort()
            val_idx.sort()
            train_datasets[pair_key] = Subset(ds_full, train_idx)
            val_datasets[pair_key] = Subset(ds_full, val_idx)

    return train_datasets, val_datasets, {}


def get_paired_tri_datasets_sachin(dataset_name, device='cpu', objects=None,
                                   config_path=None, pair_strides=None):
    """Top-level loader for `dataset_sachin` (or any sibling in the same yaml).

    Loads the Sachin-variant config from config/datasets_config_sachin.yaml
    (or override via config_path), then dispatches to the per-pair-aware
    builder. Returns the same (train, val, test) triple as the upstream
    get_all_contrastive_datasets but only for the cross_modal_tri type.
    """
    if config_path is None:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = os.path.join(repo_root, 'config', 'datasets_config_sachin.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if dataset_name not in config:
        raise KeyError(
            f"'{dataset_name}' not found in {config_path}; "
            f"available entries: {list(config.keys())}"
        )

    dataset_details = config[dataset_name]
    if dataset_details.get('dataset_type') != 'cross_modal_tri':
        raise ValueError(
            f"datasets_sachin currently only supports dataset_type='cross_modal_tri'; "
            f"'{dataset_name}' is type '{dataset_details.get('dataset_type')}'."
        )

    return get_cross_modal_tri_datasets_sachin(
        dataset_name, dataset_details, device=device, objects=objects,
        pair_strides=pair_strides,
    )
