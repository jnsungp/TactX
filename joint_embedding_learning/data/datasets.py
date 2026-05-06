from torch.utils.data import Dataset, random_split, Subset
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


def get_all_contrastive_datasets(dataset_name, data_split = 'train', device = 'cpu'):
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
    else:
        raise ValueError('Dataset type not recognized')

