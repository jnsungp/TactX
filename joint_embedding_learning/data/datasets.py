from torch.utils.data import Dataset, random_split, Subset, ConcatDataset, Sampler
import torchvision
from torchvision import transforms
from joint_embedding_learning.utils.dataset_management import sort_order, get_images_from_full_data, get_transformations
from simclr.modules.transformations import TransformsSimCLR
import torch
import glob as glob
import os
import re
import yaml
import csv
from collections import defaultdict


_EP_FR_RE = re.compile(r'ep(\d+)_fr(\d+)\.pt$')


def _parse_ep_fr(path):
    """Parse 'ep0000_fr0032.pt' style filenames. Returns (episode_id, frame_id) or (-1, -1)."""
    m = _EP_FR_RE.search(os.path.basename(path))
    if m is None:
        return -1, -1
    return int(m.group(1)), int(m.group(2))

class CMTJESimCLR(Dataset):
    def __init__(self, root_dir_bubbles, root_dir_gelslim, selected_tools, bubbles_transforms=None, gelslim_transform=None, angle_start = 0, angle_skip = 1, side = 'both', device = 'cpu', padding = False, difference = True):
        self.side = side
        self.bubbles_files = []
        self.gelslim_files = []
        self.tool_per_sample = []
        self.idx_per_tool = []

        for i, tool in enumerate(selected_tools):
            len_tool_dataset = len(glob.glob(os.path.join(root_dir_bubbles, 'bubble_style_transfer_dataset_bubbles_' + tool, '*.pt')))
            self.tool_per_sample += [i] * len_tool_dataset
            self.idx_per_tool += list(range(len_tool_dataset))
            self.bubbles_files += sorted(glob.glob(os.path.join(root_dir_bubbles, 'bubble_style_transfer_dataset_bubbles_' + tool, '*.pt')), key=sort_order)
            self.gelslim_files += sorted(glob.glob(os.path.join(root_dir_gelslim, 'gelslim_style_transfer_dataset_gelslim_' + tool, '*.pt')), key=sort_order)
        
        files_idx = list(range(angle_start, len(self.bubbles_files), angle_skip))
        self.tool_per_sample = [self.tool_per_sample[i] for i in files_idx]
        self.bubbles_files = [self.bubbles_files[i] for i in files_idx]
        self.gelslim_files = [self.gelslim_files[i] for i in files_idx]
        self.bubblest_transforms = bubbles_transforms
        self.gelslim_transform = gelslim_transform
        self.device = device
        self.padding = padding
        self.difference = difference

    def __len__(self):        
        return len(self.bubbles_files)

    def __getitem__(self, idx):
        # csv_file = '/home/samanta/tactile_style_transfer/new_processed_data/training_data_info.csv'
        # # Check if the CSV file already has content
        # file_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0

        bubbles_data = torch.load(self.bubbles_files[idx], map_location=self.device, weights_only=False)
        gelslim_data = torch.load(self.gelslim_files[idx], map_location=self.device, weights_only=False)
        # print('Index on dataset:', idx)

        # with open(csv_file, 'a', newline='') as file:
        #     writer = csv.writer(file)

        #     if not file_exists:
        #         writer.writerow(['Tool_name', 'Data_idx', 'x', 'y', 'theta'])

        #     tool_name = os.path.basename(os.path.dirname(self.gelslim_files[idx])).split('gelslim_style_transfer_dataset_gelslim_')[-1]
        #     data_idx = os.path.basename(self.gelslim_files[idx]).split('_')[-1].split('.pt')[0]
        #     x = gelslim_data['x']
        #     y = gelslim_data['y']
        #     theta = gelslim_data['theta']
        #     writer.writerow([tool_name, data_idx, x.item(), y.item(), theta.item()])

        bubbles_img, gelslim_diff, _, _ = get_images_from_full_data(bubbles_data, gelslim_data, bubbles_transform = self.bubblest_transforms, gelslim_transform = self.gelslim_transform, side = self.side, padding = self.padding, difference = self.difference)
        bubbles_img = bubbles_img.repeat(3, 1, 1)
        # import pdb; pdb.set_trace()
        return [bubbles_img, gelslim_diff], [torch.tensor(self.tool_per_sample[idx]).to(self.device), gelslim_data['theta'].to(self.device), gelslim_data['x'].to(self.device), gelslim_data['y'].to(self.device)] 
    
def get_tactile_datasets(dataset_name, dataset_details, data_split = 'train', device = 'cpu'):
    root_dir_bubbles = dataset_details['bubbles_path']
    root_dir_gelslim = dataset_details['gelslim_path']
    training_tools = dataset_details['tools']['training_tools']
    test_tools = dataset_details['tools']['test_tools']
    angle_skip = dataset_details['num_angles_skip']
    padding = dataset_details['padding']
    difference = dataset_details['difference']
    bubbles_transforms, gelslim_transform = get_transformations(dataset_name, device=device)
    
    datasets_right_train = []
    datasets_left_train = []
    datasets_right_unseen_grasps_val = []
    datasets_left_unseen_grasps_val = []
    datasets_right_unseen_grasps_test = []
    datasets_left_unseen_grasps_test = []
    datasets_right_unseen_tools_val = []
    datasets_left_unseen_tools_val = []
    datasets_right_unseen_tools_test = []
    datasets_left_unseen_tools_test = []

    for i in range(angle_skip):
        dataset_right = CMTJESimCLR(root_dir_bubbles, root_dir_gelslim, training_tools, bubbles_transforms=bubbles_transforms, gelslim_transform=gelslim_transform, angle_start = i, angle_skip=angle_skip, side='right', device=device, padding=padding, difference=difference)
        dataset_left = CMTJESimCLR(root_dir_bubbles, root_dir_gelslim, training_tools, bubbles_transforms=bubbles_transforms, gelslim_transform=gelslim_transform, angle_start = i, angle_skip=angle_skip, side='left', device=device, padding=padding, difference=difference)
        train_len = int(0.8 * len(dataset_right))
        unseen_len = len(dataset_right) - train_len
        dataset_right_train, dataset_right_unseen_grasps = random_split(dataset_right, [train_len, unseen_len], generator=torch.Generator().manual_seed(0))
        dataset_left_train, dataset_left_unseen_grasps = random_split(dataset_left, [train_len, unseen_len], generator=torch.Generator().manual_seed(0))

        datasets_right_train += [dataset_right_train]
        datasets_left_train += [dataset_left_train]

        val_len = int(0.5 * len(dataset_right_unseen_grasps))
        test_len = len(dataset_right_unseen_grasps) - val_len
        dataset_right_unseen_grasps_val, dataset_right_unseen_grasps_test = random_split(dataset_right_unseen_grasps, [val_len, test_len], generator=torch.Generator().manual_seed(0))
        dataset_left_unseen_grasps_val, dataset_left_unseen_grasps_test = random_split(dataset_left_unseen_grasps, [val_len, test_len], generator=torch.Generator().manual_seed(0))
        datasets_right_unseen_grasps_val += [dataset_right_unseen_grasps_val]
        datasets_left_unseen_grasps_val += [dataset_left_unseen_grasps_val]
        datasets_right_unseen_grasps_test += [dataset_right_unseen_grasps_test]
        datasets_left_unseen_grasps_test += [dataset_left_unseen_grasps_test]

        dataset_right_unseen_tools = CMTJESimCLR(root_dir_bubbles, root_dir_gelslim, test_tools, bubbles_transforms=bubbles_transforms, gelslim_transform=gelslim_transform, angle_start = i, angle_skip=angle_skip, side='right', device=device, padding=padding, difference=difference)
        dataset_left_unseen_tools = CMTJESimCLR(root_dir_bubbles, root_dir_gelslim, test_tools, bubbles_transforms=bubbles_transforms, gelslim_transform=gelslim_transform, angle_start = i, angle_skip=angle_skip, side='left', device=device, padding=padding, difference=difference)
        val_len = int(0.5 * len(dataset_right_unseen_tools))
        test_len = len(dataset_right_unseen_tools) - val_len
        dataset_right_unseen_tools_val, dataset_right_unseen_tools_test = random_split(dataset_right_unseen_tools, [val_len, test_len], generator=torch.Generator().manual_seed(0))
        dataset_left_unseen_tools_val, dataset_left_unseen_tools_test = random_split(dataset_left_unseen_tools, [val_len, test_len], generator=torch.Generator().manual_seed(0))
        datasets_right_unseen_tools_val += [dataset_right_unseen_tools_val]
        datasets_left_unseen_tools_val += [dataset_left_unseen_tools_val]
        datasets_right_unseen_tools_test += [dataset_right_unseen_tools_test]
        datasets_left_unseen_tools_test += [dataset_left_unseen_tools_test]

    datasets_train = datasets_right_train + datasets_left_train
    datasets_unseen_grasps_val = datasets_right_unseen_grasps_val + datasets_left_unseen_grasps_val
    datasets_unseen_grasps_test = datasets_right_unseen_grasps_test + datasets_left_unseen_grasps_test
    datasets_unseen_tools_val = datasets_right_unseen_tools_val + datasets_left_unseen_tools_val
    datasets_unseen_tools_test = datasets_right_unseen_tools_test + datasets_left_unseen_tools_test

    if data_split == 'train':
        return datasets_train, datasets_unseen_grasps_val, datasets_unseen_tools_val
    else:
        return datasets_train, datasets_unseen_grasps_test, datasets_unseen_tools_test
    
def get_stl_10_datasets(dataset_details, data_split = 'train'):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'datasets')
    img_size = dataset_details['image_size']
    train_percent = dataset_details['train_percent']

    if data_split == 'train':
        train_dataset = torchvision.datasets.STL10(path, split='unlabeled', download=True, transform=TransformsSimCLR(size=img_size))
        if train_percent == 1.0:
            train_len = int(0.9 * len(train_dataset))
            val_len = int(0.5*(len(train_dataset) - train_len))
            test_len = len(train_dataset) - train_len - val_len
            _ , val_dataset, test_dataset = random_split(train_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(0))
        else:
            train_len = int(train_percent * len(train_dataset))
            val_len = int(0.5 * (len(train_dataset) - train_len))
            test_len = len(train_dataset) - train_len - val_len
            train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(0))
    elif data_split == 'test':
        train_dataset = torchvision.datasets.STL10(
                                                        path,
                                                        split="train",
                                                        download=True,
                                                        transform=TransformsSimCLR(size=img_size).test_transform,
                                                    )
        test_dataset = torchvision.datasets.STL10(
                                                            path,
                                                            split="test",
                                                            download=True,
                                                            transform=TransformsSimCLR(size=img_size).test_transform,
                                                    )
        val_dataset = []
    return [train_dataset], [val_dataset], [test_dataset]
    
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
    """

    def __init__(self, daimon_path, eflesh_path, selected_tools,
                 daimon_transforms=None, eflesh_mean=None, eflesh_std=None,
                 daimon_channels=3, device='cpu'):
        self.daimon_files = []
        self.eflesh_files = []
        self.tool_per_sample = []
        self.episode_per_sample = []
        self.frame_per_sample = []

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
            for f in daimon_tool_files:
                ep, fr = _parse_ep_fr(f)
                self.episode_per_sample.append(ep)
                self.frame_per_sample.append(fr)

        self.daimon_transforms = daimon_transforms
        self.eflesh_mean = torch.tensor(eflesh_mean, dtype=torch.float32) if eflesh_mean is not None else None
        self.eflesh_std = torch.tensor(eflesh_std, dtype=torch.float32) if eflesh_std is not None else None
        self.daimon_channels = daimon_channels
        self.device = device

    def __len__(self):
        return len(self.daimon_files)

    def __getitem__(self, idx):
        daimon_data = torch.load(self.daimon_files[idx], map_location=self.device, weights_only=False)
        eflesh_data = torch.load(self.eflesh_files[idx], map_location=self.device, weights_only=False)

        # --- Daimon image processing ---
        daimon_img = daimon_data['image'].float()  # (C, H, W) or (H, W)
        if daimon_img.dim() == 2:
            daimon_img = daimon_img.unsqueeze(0)  # (1, H, W)
        if self.daimon_channels == 3 and daimon_img.shape[0] == 1:
            daimon_img = daimon_img.repeat(3, 1, 1)  # (3, H, W)

        if self.daimon_transforms:
            daimon_img = self.daimon_transforms(daimon_img)

        # --- eFlesh magnetic processing ---
        eflesh_vec = eflesh_data['magnetic'].float()  # (15,)
        if self.eflesh_mean is not None and self.eflesh_std is not None:
            eflesh_vec = (eflesh_vec - self.eflesh_mean.to(eflesh_vec.device)) / (self.eflesh_std.to(eflesh_vec.device) + 1e-8)

        tool_id = torch.tensor(self.tool_per_sample[idx], dtype=torch.long).to(self.device)
        episode_id = torch.tensor(self.episode_per_sample[idx], dtype=torch.long).to(self.device)
        frame_id = torch.tensor(self.frame_per_sample[idx], dtype=torch.long).to(self.device)

        return [daimon_img, eflesh_vec], [tool_id, episode_id, frame_id]


def get_cross_modal_datasets(dataset_name, dataset_details, data_split='train', device='cpu'):
    """Build train/val datasets for cross-modal (Daimon + eFlesh) contrastive learning."""
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

    if val_tools:
        # Dedicated val set exists → use 100% of training data, val from val_tools
        dataset_train = CrossModalDataset(
            daimon_path, eflesh_path, training_tools, **ds_kwargs
        )
        dataset_val = CrossModalDataset(
            daimon_path, eflesh_path, val_tools, **ds_kwargs
        )
        unseen_grasps_val_list = [dataset_val]
        unseen_grasps_test_list = [dataset_val]
    else:
        # No dedicated val set → split training data 80/20
        dataset_train_full = CrossModalDataset(
            daimon_path, eflesh_path, training_tools, **ds_kwargs
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


class CrossModalPairDataset(Dataset):
    """
    Single-pair dataset for tri-modal training (raw-tensor .pt format).

    Directory layout (per pair root):
        pair_root/sensor_a/ep****_fr****.pt
        pair_root/sensor_b/ep****_fr****.pt

    Each .pt file contains a *raw tensor* (NOT a dict):
        Daimon  : (3, 240, 320)  float32
        eFlesh  : (15,)          float32
        Flexi   : (12, 16)       float32

    Pairing rule: exact basename match (ep0000_fr0001.pt in sensor_a/ <-> same
    name in sensor_b/). Files that exist on only one side are dropped.
    """

    def __init__(self, pair_root, sensor_a, sensor_b,
                 transforms_a=None, transforms_b=None,
                 mean_a=None, std_a=None, mean_b=None, std_b=None,
                 device='cpu', objects=None, frame_stride=1):
        a_dir = os.path.join(pair_root, sensor_a)
        b_dir = os.path.join(pair_root, sensor_b)

        if objects is not None:
            # Per-object layout: sensor/{object}/ep****_fr****.pt
            a_files = {}
            b_files = {}
            for obj in objects:
                for p in glob.glob(os.path.join(a_dir, obj, '*.pt')):
                    a_files[f"{obj}/{os.path.basename(p)}"] = p
                for p in glob.glob(os.path.join(b_dir, obj, '*.pt')):
                    b_files[f"{obj}/{os.path.basename(p)}"] = p
        else:
            # Flat layout: sensor/ep****_fr****.pt
            a_files = {os.path.basename(p): p
                       for p in glob.glob(os.path.join(a_dir, '*.pt'))}
            b_files = {os.path.basename(p): p
                       for p in glob.glob(os.path.join(b_dir, '*.pt'))}
        common = sorted(set(a_files) & set(b_files), key=sort_order)

        self.sensor_a = sensor_a
        self.sensor_b = sensor_b
        self.files_a = [a_files[n] for n in common]
        self.files_b = [b_files[n] for n in common]
        self.episode_per_sample = []
        self.frame_per_sample = []
        # object id per sample: parsed from the "object/" key prefix used in
        # the per-object layout (flat layout -> all 0). Needed by CWCL so soft
        # positives are gated to the same object (episode ids are NOT unique
        # across objects — ep0000 exists for every object).
        self.object_per_sample = []
        self.object_names = []
        _obj_to_id = {}
        for n in common:
            ep, fr = _parse_ep_fr(n)
            self.episode_per_sample.append(ep)
            self.frame_per_sample.append(fr)
            obj = n.split('/')[0] if '/' in n else '_flat_'
            if obj not in _obj_to_id:
                _obj_to_id[obj] = len(self.object_names)
                self.object_names.append(obj)
            self.object_per_sample.append(_obj_to_id[obj])

        # Per-episode temporal subsampling: keep every `frame_stride`-th frame
        # within each (object, episode). Used to down-sample a higher-rate
        # pair (e.g. ef = eFlesh+FlexiTac) so the three pairs have comparable
        # temporal density / dataset size.
        if frame_stride and frame_stride > 1:
            by_ep = defaultdict(list)
            for i in range(len(self.files_a)):
                by_ep[(self.object_per_sample[i],
                       self.episode_per_sample[i])].append(i)
            keep = []
            for idxs in by_ep.values():
                idxs.sort(key=lambda i: self.frame_per_sample[i])
                keep.extend(idxs[::frame_stride])
            keep.sort()
            self.files_a = [self.files_a[i] for i in keep]
            self.files_b = [self.files_b[i] for i in keep]
            self.episode_per_sample = [self.episode_per_sample[i] for i in keep]
            self.frame_per_sample = [self.frame_per_sample[i] for i in keep]
            self.object_per_sample = [self.object_per_sample[i] for i in keep]

        self.transforms_a = transforms_a
        self.transforms_b = transforms_b
        self.mean_a = torch.tensor(mean_a, dtype=torch.float32) if mean_a is not None else None
        self.std_a = torch.tensor(std_a, dtype=torch.float32) if std_a is not None else None
        self.mean_b = torch.tensor(mean_b, dtype=torch.float32) if mean_b is not None else None
        self.std_b = torch.tensor(std_b, dtype=torch.float32) if std_b is not None else None
        self.device = device

    def __len__(self):
        return len(self.files_a)

    @staticmethod
    def _apply_zscore(x, mean, std):
        if mean is None or std is None:
            return x
        m = mean.to(x.device)
        s = std.to(x.device)
        # broadcast over the leading dims; eFlesh: (15,) over (15,);
        # Daimon: (3,) over (3, H, W) requires reshape to (3, 1, 1).
        if x.dim() >= 2 and m.dim() == 1 and m.shape[0] == x.shape[0]:
            m = m.view(-1, *([1] * (x.dim() - 1)))
            s = s.view(-1, *([1] * (x.dim() - 1)))
        return (x - m) / (s + 1e-8)

    def _load_one(self, path, transforms_, mean, std):
        x = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(x, dict):
            # backward-compat: also accept dict-style files if encountered.
            if 'image' in x: x = x['image']
            elif 'magnetic' in x: x = x['magnetic']
            elif 'flexi' in x or 'flexitac' in x:
                x = x.get('flexi', x.get('flexitac'))
            else:
                raise ValueError(f"Unknown dict keys in {path}: {list(x.keys())}")
        x = x.float()
        x = self._apply_zscore(x, mean, std)
        if transforms_ is not None:
            x = transforms_(x)
        return x

    def __getitem__(self, idx):
        x_a = self._load_one(self.files_a[idx], self.transforms_a,
                             self.mean_a, self.std_a)
        x_b = self._load_one(self.files_b[idx], self.transforms_b,
                             self.mean_b, self.std_b)
        object_id = torch.tensor(self.object_per_sample[idx], dtype=torch.long).to(self.device)
        episode_id = torch.tensor(self.episode_per_sample[idx], dtype=torch.long).to(self.device)
        frame_id = torch.tensor(self.frame_per_sample[idx], dtype=torch.long).to(self.device)
        # label triple: [object_id, episode_id, frame_id]. Consumers that only
        # used [episode, frame] ignore labels entirely, so this is additive.
        return [x_a, x_b], [object_id, episode_id, frame_id]


class EpisodeChunkBatchSampler(Sampler):
    """Batch sampler that builds each batch from contiguous frame-chunks of a
    few episodes, so same-episode frames co-occur within the batch.

    This is required for CWCL: its soft positives are same-(object,episode)
    frames weighted by frame distance, and plain random shuffling almost never
    co-locates them in a batch (W collapses to identity → CWCL ≈ InfoNCE).

    Each epoch:
      - group sample indices by (object_id, episode_id), sort within by frame
      - cut each group into contiguous chunks of `chunk_size` frames
      - shuffle chunk order; pack `batch_size // chunk_size` chunks per batch

    So every batch contains several contiguous frame-runs — within a run CWCL
    sees graded soft positives, across runs they are negatives.

    Works on a CrossModalPairDataset or a Subset of one (e.g. from
    random_split). Batch sizes vary slightly (partial trailing chunks); CWCL
    is batch-size agnostic so that is fine.
    """

    def __init__(self, dataset, batch_size, chunk_size=8, drop_last=True, seed=0):
        self.batch_size = int(batch_size)
        self.chunk_size = max(1, int(chunk_size))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        obj, ep, fr = self._extract_meta(dataset)
        groups = defaultdict(list)
        for i in range(len(obj)):
            groups[(obj[i], ep[i])].append(i)

        self.chunks = []
        for idxs in groups.values():
            idxs.sort(key=lambda i: fr[i])
            for s in range(0, len(idxs), self.chunk_size):
                self.chunks.append(idxs[s:s + self.chunk_size])

        self.chunks_per_batch = max(1, self.batch_size // self.chunk_size)

    @staticmethod
    def _extract_meta(dataset):
        """Return (object_ids, episode_ids, frame_ids) as local-indexed lists,
        transparently unwrapping a Subset."""
        if isinstance(dataset, Subset):
            base, idxs = dataset.dataset, dataset.indices
            return ([base.object_per_sample[i] for i in idxs],
                    [base.episode_per_sample[i] for i in idxs],
                    [base.frame_per_sample[i] for i in idxs])
        return (list(dataset.object_per_sample),
                list(dataset.episode_per_sample),
                list(dataset.frame_per_sample))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        n_full = len(self.chunks) // self.chunks_per_batch
        rem = len(self.chunks) % self.chunks_per_batch
        return n_full + (1 if (rem and not self.drop_last) else 0)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.chunks), generator=g).tolist()
        batch, cnt = [], 0
        for ci in order:
            batch.extend(self.chunks[ci])
            cnt += 1
            if cnt == self.chunks_per_batch:
                yield batch
                batch, cnt = [], 0
        if batch and not self.drop_last:
            yield batch
        self.epoch += 1


class DynamicSelfDataset(Dataset):
    """
    Single-sensor (unpaired) dataset for Stage 1 dynamic self-recon pretraining.

    Directory layout (flat):
        sensor_root/ep****_fr****.pt

    Each .pt is a raw tensor (see CrossModalPairDataset for sensor shapes).
    No pairing across modalities; each modality has its own dataset.
    """

    def __init__(self, sensor_root, sensor_name,
                 transform=None, mean=None, std=None, device='cpu', objects=None):
        files = sorted(glob.glob(os.path.join(sensor_root, '*.pt')),
                       key=_parse_ep_fr)
        # Fall back to one extra level ({object}/ep****_fr****.pt layout).
        ep_id_offsets = [0] * len(files)
        if not files:
            subdirs = sorted(d for d in glob.glob(os.path.join(sensor_root, '*'))
                             if os.path.isdir(d))
            if objects is not None:
                allow = set(objects)
                subdirs = [d for d in subdirs if os.path.basename(d) in allow]
            files = []
            ep_id_offsets = []
            for obj_idx, subdir in enumerate(subdirs):
                obj_files = sorted(glob.glob(os.path.join(subdir, '*.pt')),
                                   key=_parse_ep_fr)
                files.extend(obj_files)
                ep_id_offsets.extend([obj_idx * 100000] * len(obj_files))

        self.sensor_name = sensor_name
        self.files = files
        self.episode_per_sample = []
        self.frame_per_sample = []
        for p, offset in zip(files, ep_id_offsets):
            ep, fr = _parse_ep_fr(p)
            self.episode_per_sample.append(ep + offset)
            self.frame_per_sample.append(fr)

        # Mark the minimum frame_id per episode as no-contact (padding/reference frame).
        ep_min_fr = defaultdict(lambda: float('inf'))
        for ep, fr in zip(self.episode_per_sample, self.frame_per_sample):
            if fr < ep_min_fr[ep]:
                ep_min_fr[ep] = fr
        self.is_nocontact_per_sample = [
            fr == ep_min_fr[ep]
            for ep, fr in zip(self.episode_per_sample, self.frame_per_sample)
        ]

        self.transform = transform
        self.mean = torch.tensor(mean, dtype=torch.float32) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float32) if std is not None else None
        self.device = device

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _apply_zscore(x, mean, std):
        if mean is None or std is None:
            return x
        m = mean.to(x.device)
        s = std.to(x.device)
        if x.dim() >= 2 and m.dim() == 1 and m.shape[0] == x.shape[0]:
            m = m.view(-1, *([1] * (x.dim() - 1)))
            s = s.view(-1, *([1] * (x.dim() - 1)))
        return (x - m) / (s + 1e-8)

    def __getitem__(self, idx):
        path = self.files[idx]
        x = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(x, dict):
            if 'image' in x: x = x['image']
            elif 'magnetic' in x: x = x['magnetic']
            elif 'flexi' in x or 'flexitac' in x:
                x = x.get('flexi', x.get('flexitac'))
            else:
                raise ValueError(f"Unknown dict keys in {path}: {list(x.keys())}")
        x = x.float()
        x = self._apply_zscore(x, self.mean, self.std)
        if self.transform is not None:
            x = self.transform(x)
        episode_id = torch.tensor(self.episode_per_sample[idx], dtype=torch.long).to(self.device)
        frame_id = torch.tensor(self.frame_per_sample[idx], dtype=torch.long).to(self.device)
        is_nocontact = torch.tensor(self.is_nocontact_per_sample[idx], dtype=torch.bool)
        return x, [episode_id, frame_id, is_nocontact]


class TemporalPairDataset(Dataset):
    """Consecutive (frame_t, frame_{t+1}) pairs from the same episode.

    Built from a DynamicSelfDataset. Adjacent entries in the sorted file list
    that share the same episode_id are treated as consecutive frames.
    Used to compute L_slow = ||mu_t - mu_{t+1}||^2 during training.
    """

    def __init__(self, base):
        self.base = base
        eps = base.episode_per_sample
        self.pairs = [
            (i, i + 1) for i in range(len(eps) - 1) if eps[i] == eps[i + 1]
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        x_t, _ = self.base[i]
        x_t1, _ = self.base[j]
        return x_t, x_t1


def _build_sensor_transforms(dataset_details, sensor_name, image_size):
    """Build (transforms, mean, std) for a sensor from the dataset_details yaml.

    Conventions:
      - daimon: optional ['resize', 'normalize']. Normalize uses
        daimon_stats.mean / std (length 3 for the 3-channel input).
        We do z-score via tensor stats (same as torchvision Normalize).
      - eflesh: optional ['normalize']. Uses eflesh_stats (length 15).
      - flexi : optional ['normalize']. Uses flexitac_stats (scalar mean/std).
    """
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
                # Allow scalar or list; broadcast scalar to a single-element tensor.
                m = stats['mean']
                s = stats['std']
                mean = m if isinstance(m, (list, tuple)) else [float(m)]
                std = s if isinstance(s, (list, tuple)) else [float(s)]
    else:
        raise ValueError(f"Unknown sensor: {sensor_name}")

    transform = transforms.Compose(transforms_list) if transforms_list else None
    return transform, mean, std


def get_cross_modal_tri_datasets(dataset_name, dataset_details, data_split='train', device='cpu', objects=None):
    """Build 80/20 train/val datasets for tri-modal (Daimon + eFlesh + FlexiTac).

    Args:
        objects: optional list of object names to include (e.g. ['cylinder', 'plug']).
                 Requires per-object directory layout under each sensor dir.
                 If None, uses the flat layout (all files directly under sensor dir).

    Returns:
        train_datasets : dict {'de': ds, 'ef': ds, 'fd': ds}
        val_datasets   : dict {'de': ds_val, 'ef': ds_val, 'fd': ds_val}
        empty_test     : {}
    """
    data_root = dataset_details['data_root']
    image_size = dataset_details.get('image_size', 224)
    pairs = dataset_details['pairs']
    train_frac = float(dataset_details.get('train_frac', 0.8))

    sensor_cache = {}
    def _trans(sensor):
        if sensor not in sensor_cache:
            sensor_cache[sensor] = _build_sensor_transforms(dataset_details, sensor, image_size)
        return sensor_cache[sensor]

    train_datasets = {}
    val_datasets = {}

    for pair_key, pair_cfg in pairs.items():
        sensor_a = pair_cfg['sensor_a']
        sensor_b = pair_cfg['sensor_b']
        pair_root = os.path.join(data_root, pair_cfg['root'])

        ta, ma, sa = _trans(sensor_a)
        tb, mb, sb = _trans(sensor_b)
        ds_full = CrossModalPairDataset(
            pair_root, sensor_a, sensor_b,
            transforms_a=ta, transforms_b=tb,
            mean_a=ma, std_a=sa, mean_b=mb, std_b=sb,
            device=device, objects=objects,
        )

        n = len(ds_full)
        train_len = int(train_frac * n)
        val_len = n - train_len
        ds_train, ds_val = random_split(
            ds_full, [train_len, val_len],
            generator=torch.Generator().manual_seed(0),
        )

        train_datasets[pair_key] = ds_train
        val_datasets[pair_key] = ds_val

    return train_datasets, val_datasets, {}


def get_dynamic_self_datasets(dataset_name, dataset_details, data_split='train', device='cpu', objects=None):
    """Build per-modality datasets for Stage 1 dynamic self-recon pretrain.

    Layout: data_root/{daimon|eflesh|flexi}/ep****_fr****.pt (flat, unpaired)

    Returns:
        train_datasets : dict {'D': ds_D, 'E': ds_E, 'F': ds_F}
        val_datasets   : dict {'D': ds_D_val, 'E': ds_E_val, 'F': ds_F_val}
        empty_test     : {}
    """
    # sensor_roots: per-sensor override (e.g. different pair dirs per modality).
    # Falls back to data_root for sensors not listed.
    sensor_roots_cfg = dataset_details.get('sensor_roots', {})
    data_root = dataset_details.get('data_root', None)
    image_size = dataset_details.get('image_size', 224)
    sensors = dataset_details.get('sensors', {'D': 'daimon', 'E': 'eflesh', 'F': 'flexi'})
    train_frac = float(dataset_details.get('train_frac', 0.8))

    train_datasets = {}
    val_datasets = {}

    for key, sensor_name in sensors.items():
        transform, mean, std = _build_sensor_transforms(dataset_details, sensor_name, image_size)
        root = sensor_roots_cfg.get(key, data_root)
        data_roots = [root] if isinstance(root, str) else list(root)
        per_root = [
            DynamicSelfDataset(
                os.path.join(dr, sensor_name), sensor_name,
                transform=transform, mean=mean, std=std,
                device=device, objects=objects,
            )
            for dr in data_roots
        ]
        ds_full = per_root[0] if len(per_root) == 1 else ConcatDataset(per_root)

        use_ep_split = (bool(dataset_details.get('episode_val_split', False))
                        and hasattr(ds_full, 'episode_per_sample'))
        if use_ep_split:
            groups = defaultdict(list)
            for i, ep_full in enumerate(ds_full.episode_per_sample):
                obj_id = ep_full // 100000
                ep_id  = ep_full % 100000
                groups[(obj_id, ep_id)].append(i)
            train_idx, val_idx = [], []
            explicit_val_eps = dataset_details.get('val_episode_ids', None)
            if explicit_val_eps is not None:
                val_ep_set = set(int(e) for e in explicit_val_eps)
                for (obj_id, ep_id), idxs in groups.items():
                    if ep_id in val_ep_set:
                        val_idx.extend(idxs)
                    else:
                        train_idx.extend(idxs)
                split_label = f"(val eps = {sorted(val_ep_set)})"
            else:
                by_obj = defaultdict(list)
                for gk in groups:
                    by_obj[gk[0]].append(gk)
                val_stride = max(2, round(1.0 / max(1e-6, 1.0 - train_frac)))
                for obj_id in sorted(by_obj):
                    for j, gk in enumerate(sorted(by_obj[obj_id])):
                        (val_idx if j % val_stride == 0 else train_idx).extend(groups[gk])
                split_label = f"(val=every {val_stride}th ep per object)"
            train_idx.sort()
            val_idx.sort()
            train_datasets[key] = Subset(ds_full, train_idx)
            val_datasets[key]   = Subset(ds_full, val_idx)
            print(f"  [{key}/{sensor_name}] episode-level split: "
                  f"{len(train_idx)} train / {len(val_idx)} val "
                  f"{split_label}")
        else:
            n = len(ds_full)
            train_len = int(train_frac * n)
            val_len = n - train_len
            ds_train, ds_val = random_split(
                ds_full, [train_len, val_len],
                generator=torch.Generator().manual_seed(0),
            )
            train_datasets[key] = ds_train
            val_datasets[key]   = ds_val

    return train_datasets, val_datasets, {}


def get_temporal_pair_datasets(dataset_name, device='cpu', objects=None):
    """Per-modality TemporalPairDataset dict for L_slow training.

    Same config lookup as get_dynamic_self_datasets but returns consecutive
    (frame_t, frame_{t+1}) pair datasets rather than shuffled singletons.
    Only the training split is returned (L_slow is not validated).

    Returns:
        dict {'D': TemporalPairDataset, 'E': ..., 'F': ...}
    """
    dataset_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config', 'datasets_config.yaml',
    )
    with open(dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    dataset_details = config[dataset_name]

    sensor_roots_cfg = dataset_details.get('sensor_roots', {})
    data_root = dataset_details.get('data_root', None)
    image_size = dataset_details.get('image_size', 224)
    sensors = dataset_details.get('sensors', {'D': 'daimon', 'E': 'eflesh', 'F': 'flexi'})
    train_frac = float(dataset_details.get('train_frac', 0.8))
    # Mirror get_dynamic_self_datasets's episode-level split so L_slow only
    # walks consecutive pairs whose BOTH frames lie in train episodes — val
    # episodes (whole groups) are dropped at pair-build time.
    use_ep_split = bool(dataset_details.get('episode_val_split', False))
    explicit_val_eps = dataset_details.get('val_episode_ids', None)

    result = {}
    for key, sensor_name in sensors.items():
        transform, mean, std = _build_sensor_transforms(dataset_details, sensor_name, image_size)
        root = sensor_roots_cfg.get(key, data_root)
        data_roots = [root] if isinstance(root, str) else list(root)
        pair_datasets = []
        for dr in data_roots:
            base = DynamicSelfDataset(
                os.path.join(dr, sensor_name), sensor_name,
                transform=transform, mean=mean, std=std,
                device=device, objects=objects,
            )
            tp = TemporalPairDataset(base)
            if use_ep_split:
                groups = defaultdict(list)
                for i, ep_full in enumerate(base.episode_per_sample):
                    obj_id = ep_full // 100000
                    ep_id = ep_full % 100000
                    groups[(obj_id, ep_id)].append(i)
                train_set = set()
                if explicit_val_eps is not None:
                    val_ep_set = set(int(e) for e in explicit_val_eps)
                    for (obj_id, ep_id), idxs in groups.items():
                        if ep_id not in val_ep_set:
                            train_set.update(idxs)
                else:
                    by_obj = defaultdict(list)
                    for gk in groups:
                        by_obj[gk[0]].append(gk)
                    val_stride = max(2, round(1.0 / max(1e-6, 1.0 - train_frac)))
                    for obj_id in sorted(by_obj):
                        for j, gk in enumerate(sorted(by_obj[obj_id])):
                            if j % val_stride != 0:
                                train_set.update(groups[gk])
                n_total = len(tp.pairs)
                tp.pairs = [(i, j) for (i, j) in tp.pairs
                            if i in train_set and j in train_set]
                print(f"  [temporal_pairs/{sensor_name}] train-episode filter: "
                      f"kept {len(tp.pairs)}/{n_total} pairs")
            pair_datasets.append(tp)
        result[key] = pair_datasets[0] if len(pair_datasets) == 1 else ConcatDataset(pair_datasets)
        print(f"[temporal_pairs] {sensor_name}: {len(result[key])} consecutive pairs")
    return result


def get_all_contrastive_datasets(dataset_name, data_split='train', device='cpu', objects=None):
    dataset_config =  os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'datasets_config.yaml')

    with open(dataset_config, 'r') as file:
        config = yaml.safe_load(file)

    dataset_details = config[dataset_name]
    dataset_type = dataset_details['dataset_type']

    if dataset_type == 'tactile':
        return get_tactile_datasets(dataset_name, dataset_details, data_split, device)
    elif dataset_type == 'stl_10':
        return get_stl_10_datasets(dataset_details, data_split)
    elif dataset_type == 'cross_modal':
        return get_cross_modal_datasets(dataset_name, dataset_details, data_split, device)
    elif dataset_type == 'cross_modal_tri':
        return get_cross_modal_tri_datasets(dataset_name, dataset_details, data_split, device, objects=objects)
    elif dataset_type == 'cross_modal_tri_dynamic':
        return get_dynamic_self_datasets(dataset_name, dataset_details, data_split, device, objects=objects)
    else:
        raise ValueError('Dataset type not recognized')

