from torch.utils.data import Dataset, random_split
from torchvision import transforms
import torch
import numpy as np
import glob as glob
import os
import re
import yaml


def sort_order(filename):
    return int(re.findall(r'\d+', filename)[-1])


def _parse_episode_id(filename):
    """ep0003_fr0042.pt -> 3"""
    return int(os.path.basename(filename).split("_")[0].replace("ep", ""))


def _thin_episode(eflesh_vecs, hold_factor, threshold_ratio=1.5, smooth_window=3):
    """
    Identify which frames of an episode to keep.

    Detects transitions (high frame-to-frame magnetic change) vs hold phase (low change),
    keeps all transition frames, subsamples hold frames by `hold_factor`.

    Args:
        eflesh_vecs: (T, 15) np.ndarray of magnetic vectors for one episode, time-ordered.
        hold_factor: keep 1 of every `hold_factor` hold frames.
        threshold_ratio: motion threshold = median(motion) * threshold_ratio.
        smooth_window: window size for smoothing the motion signal.

    Returns:
        List of local indices (into `eflesh_vecs`) to keep.
    """
    T = len(eflesh_vecs)
    if T <= 3:
        return list(range(T))

    diffs = np.abs(np.diff(eflesh_vecs, axis=0)).sum(axis=1)     # (T-1,)
    diffs = np.concatenate([[diffs[0]], diffs])                   # (T,)

    # Simple boxcar smooth
    k = max(1, smooth_window)
    pad = k // 2
    padded = np.pad(diffs, pad, mode='edge')
    smoothed = np.convolve(padded, np.ones(k) / k, mode='valid')

    threshold = np.median(smoothed) * threshold_ratio
    is_transition = smoothed > threshold

    keep = []
    hold_counter = 0
    for t in range(T):
        if is_transition[t]:
            keep.append(t)
            hold_counter = 0
        else:
            if hold_counter == 0:
                keep.append(t)
            hold_counter = (hold_counter + 1) % hold_factor
    return keep


class CrossModalDataset(Dataset):
    """
    Cross-modal dataset for Daimon (image) and eFlesh (magnetic) sensor pairs.

    Positive pairs: same grasp frame from both sensors.
    Daimon provides tactile images, eFlesh provides 15D magnetic readings.

    Expected .pt file format:
        Daimon: {'image': tensor (C, H, W), ...}
        eFlesh: {'magnetic': tensor (15,), ...}

    Directory structure:
        daimon_path/{tool_name}/*.pt
        eflesh_path/{tool_name}/*.pt

    Optional hold-phase thinning: detect low-motion frames (trapezoid plateau) and
    subsample them to remove redundant positive pairs.
    """

    def __init__(self, daimon_path, eflesh_path, selected_tools,
                 daimon_transforms=None, eflesh_mean=None, eflesh_std=None,
                 daimon_channels=3, device='cpu',
                 thin_hold_factor=None, thin_threshold_ratio=1.5):
        self.daimon_files = []
        self.eflesh_files = []
        self.tool_per_sample = []

        for i, tool in enumerate(selected_tools):
            daimon_tool_files = sorted(
                glob.glob(os.path.join(daimon_path, tool, '*.pt')), key=sort_order
            )
            eflesh_tool_files = sorted(
                glob.glob(os.path.join(eflesh_path, tool, '*.pt')), key=sort_order
            )
            assert len(daimon_tool_files) == len(eflesh_tool_files), \
                f"Data count mismatch for tool '{tool}': daimon={len(daimon_tool_files)}, eflesh={len(eflesh_tool_files)}"

            self.daimon_files += daimon_tool_files
            self.eflesh_files += eflesh_tool_files
            self.tool_per_sample += [i] * len(daimon_tool_files)

        # Parse episode ids from filenames
        self.episode_ids = [_parse_episode_id(f) for f in self.daimon_files]

        # Thin hold phase if requested
        if thin_hold_factor is not None and thin_hold_factor > 1:
            self._thin_hold_phase(thin_hold_factor, thin_threshold_ratio)

        self.daimon_transforms = daimon_transforms
        self.eflesh_mean = torch.tensor(eflesh_mean, dtype=torch.float32) if eflesh_mean is not None else None
        self.eflesh_std = torch.tensor(eflesh_std, dtype=torch.float32) if eflesh_std is not None else None
        self.daimon_channels = daimon_channels
        self.device = device

    def _thin_hold_phase(self, hold_factor, threshold_ratio):
        """Subsample hold-phase frames per episode. Modifies file lists in place."""
        # Group indices by episode (preserving original time order within episode)
        ep_to_indices = {}
        for idx, ep in enumerate(self.episode_ids):
            ep_to_indices.setdefault(ep, []).append(idx)

        n_before = len(self.daimon_files)
        kept_indices = []
        for ep, indices in ep_to_indices.items():
            # Load eflesh vectors for this episode (small, fast)
            mags = []
            for idx in indices:
                data = torch.load(self.eflesh_files[idx], map_location='cpu', weights_only=True)
                mags.append(data['magnetic'].numpy())
            mags = np.stack(mags).astype(np.float32)  # (T, 15)

            local_keep = _thin_episode(mags, hold_factor, threshold_ratio)
            kept_indices.extend([indices[t] for t in local_keep])

        kept_indices = sorted(kept_indices)
        self.daimon_files = [self.daimon_files[i] for i in kept_indices]
        self.eflesh_files = [self.eflesh_files[i] for i in kept_indices]
        self.tool_per_sample = [self.tool_per_sample[i] for i in kept_indices]
        self.episode_ids = [self.episode_ids[i] for i in kept_indices]

        n_after = len(self.daimon_files)
        print(f"  Thinning: {n_before} -> {n_after} frames ({n_after/n_before*100:.1f}% kept, "
              f"hold_factor={hold_factor})")

    def __len__(self):
        return len(self.daimon_files)

    def __getitem__(self, idx):
        daimon_data = torch.load(self.daimon_files[idx], map_location=self.device)
        eflesh_data = torch.load(self.eflesh_files[idx], map_location=self.device)

        # --- Daimon image processing ---
        daimon_img = daimon_data['image'].float()
        if daimon_img.dim() == 2:
            daimon_img = daimon_img.unsqueeze(0)
        if self.daimon_channels == 3 and daimon_img.shape[0] == 1:
            daimon_img = daimon_img.repeat(3, 1, 1)

        if self.daimon_transforms:
            daimon_img = self.daimon_transforms(daimon_img)

        # --- eFlesh magnetic processing ---
        eflesh_vec = eflesh_data['magnetic'].float()
        if self.eflesh_mean is not None and self.eflesh_std is not None:
            eflesh_vec = (eflesh_vec - self.eflesh_mean.to(eflesh_vec.device)) / (self.eflesh_std.to(eflesh_vec.device) + 1e-8)

        label = torch.tensor(self.tool_per_sample[idx]).to(self.device)
        episode_id = torch.tensor(self.episode_ids[idx]).to(self.device)

        return [daimon_img, eflesh_vec], [label, episode_id]


def get_cross_modal_datasets(dataset_name, dataset_details, data_split='train', device='cpu',
                              thin_hold_factor=None, thin_threshold_ratio=1.5):
    """Build train/val datasets for cross-modal contrastive learning.

    thin_hold_factor: if set, applies to TRAIN set only (val stays full for unbiased metrics).
    """
    daimon_path = dataset_details['daimon_path']
    eflesh_path = dataset_details['eflesh_path']
    training_tools = dataset_details['tools']['training_tools']
    val_tools = dataset_details['tools'].get('val_tools', [])
    test_tools = dataset_details['tools'].get('test_tools', [])
    image_size = dataset_details['image_size']
    daimon_channels = dataset_details.get('daimon_channels', 3)

    # --- Daimon image transforms ---
    daimon_transforms_list = []
    daimon_transform_names = dataset_details.get('daimon_transforms', None)
    if daimon_transform_names:
        if 'resize' in daimon_transform_names:
            daimon_transforms_list.append(transforms.Resize((image_size, image_size)))
        if 'normalize' in daimon_transform_names:
            daimon_stats = dataset_details['daimon_stats']
            daimon_transforms_list.append(
                transforms.Normalize(
                    torch.tensor(daimon_stats['mean']),
                    torch.tensor(daimon_stats['std'])
                )
            )
    daimon_transforms = transforms.Compose(daimon_transforms_list) if daimon_transforms_list else None

    # --- eFlesh normalization stats ---
    eflesh_stats = dataset_details.get('eflesh_stats', None)
    eflesh_mean = eflesh_stats['mean'] if eflesh_stats else None
    eflesh_std = eflesh_stats['std'] if eflesh_stats else None

    ds_kwargs = dict(
        daimon_transforms=daimon_transforms,
        eflesh_mean=eflesh_mean, eflesh_std=eflesh_std,
        daimon_channels=daimon_channels, device=device
    )

    train_kwargs = dict(ds_kwargs)
    if thin_hold_factor is not None and thin_hold_factor > 1:
        train_kwargs['thin_hold_factor'] = thin_hold_factor
        train_kwargs['thin_threshold_ratio'] = thin_threshold_ratio

    if val_tools:
        print("Loading train set...")
        dataset_train = CrossModalDataset(
            daimon_path, eflesh_path, training_tools, **train_kwargs
        )
        print("Loading val set...")
        dataset_val = CrossModalDataset(
            daimon_path, eflesh_path, val_tools, **ds_kwargs
        )
        unseen_grasps_val_list = [dataset_val]
        unseen_grasps_test_list = [dataset_val]
    else:
        print("Loading train set...")
        dataset_train_full = CrossModalDataset(
            daimon_path, eflesh_path, training_tools, **train_kwargs
        )
        train_len = int(0.8 * len(dataset_train_full))
        unseen_len = len(dataset_train_full) - train_len
        dataset_train, dataset_unseen_grasps = random_split(
            dataset_train_full, [train_len, unseen_len],
            generator=torch.Generator().manual_seed(0)
        )
        val_len = int(0.5 * len(dataset_unseen_grasps))
        test_len = len(dataset_unseen_grasps) - val_len
        dataset_unseen_grasps_val, dataset_unseen_grasps_test = random_split(
            dataset_unseen_grasps, [val_len, test_len],
            generator=torch.Generator().manual_seed(0)
        )
        unseen_grasps_val_list = [dataset_unseen_grasps_val]
        unseen_grasps_test_list = [dataset_unseen_grasps_test]

    # --- Unseen tools dataset ---
    if test_tools:
        dataset_unseen_tools_full = CrossModalDataset(
            daimon_path, eflesh_path, test_tools, **ds_kwargs
        )
        val_unseen_tools_len = int(0.5 * len(dataset_unseen_tools_full))
        test_unseen_tools_len = len(dataset_unseen_tools_full) - val_unseen_tools_len
        dataset_unseen_tools_val, dataset_unseen_tools_test = random_split(
            dataset_unseen_tools_full, [val_unseen_tools_len, test_unseen_tools_len],
            generator=torch.Generator().manual_seed(0)
        )
        unseen_tools_val_list = [dataset_unseen_tools_val]
        unseen_tools_test_list = [dataset_unseen_tools_test]
    else:
        unseen_tools_val_list = []
        unseen_tools_test_list = []

    if data_split == 'train':
        return [dataset_train], unseen_grasps_val_list, unseen_tools_val_list
    else:
        return [dataset_train], unseen_grasps_test_list, unseen_tools_test_list


def get_all_contrastive_datasets(dataset_name, data_split='train', device='cpu',
                                  thin_hold_factor=None, thin_threshold_ratio=1.5):
    dataset_config = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'datasets_config.yaml')

    with open(dataset_config, 'r') as file:
        config = yaml.safe_load(file)

    dataset_details = config[dataset_name]
    return get_cross_modal_datasets(dataset_name, dataset_details, data_split, device,
                                     thin_hold_factor=thin_hold_factor,
                                     thin_threshold_ratio=thin_threshold_ratio)
