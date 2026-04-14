import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, SiglipImageProcessor, SiglipVisionModel

from PIL import Image
import torch.nn.functional as F


class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.eval()

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.select_feature == 'patch':
            image_features = image_forward_outs.last_hidden_state  # (B, 729, 1536)
        elif self.select_feature == 'cls_patch':
            image_features = image_forward_outs.pooler_output # (B, 1, 1536)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    @property
    def background_color(self):
        return np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vision_encoder = SiglipVisionTower("/data/pretrained_weights/siglip-so400m-patch14-384", args=None)
    vision_encoder.to(device)

    image = Image.open("test_wrist.png").convert("RGB")

    if True:
        # Padding
        image = vision_encoder.expand2square(image, tuple(int(x*255) for x in vision_encoder.image_processor.image_mean))

    image = vision_encoder.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    
    # Simulate two cameras
    image_tensor_list = []

    for _ in range(2):
        image_tensor_list.append(image)

    image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=torch.float32)

    image_features_bnc = vision_encoder(image_tensor).detach()  # n = (H // 14) * (W // 14)
    input_H, input_W = image_tensor.shape[-2:]
    output_H, output_W = (input_H // vision_encoder.config.patch_size), (input_W // vision_encoder.config.patch_size)
    image_features_bhwc = image_features_bnc.reshape(-1, output_H, output_W, vision_encoder.hidden_size)
    image_features_bchw = image_features_bhwc.permute(0, 3, 1, 2)

    # RDT put two cameras together
    rdt_features_mc = image_features_bnc.reshape(-1, vision_encoder.hidden_size).unsqueeze(0)

    print(image_features_bnc.shape)
    print(rdt_features_mc.shape)

    class TorchPCA(object):

        def __init__(self, n_components):
            self.n_components = n_components

        def fit(self, X):
            self.mean_ = X.mean(dim=0)
            unbiased = X - self.mean_.unsqueeze(0)
            U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
            self.components_ = V.T
            self.singular_values_ = S
            return self

        def transform(self, X):
            t0 = X - self.mean_.unsqueeze(0)
            projected = t0 @ self.components_.T
            return projected


    def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
        device = image_feats_list[0].device

        def flatten(tensor, target_size=None):
            if target_size is not None and fit_pca is None:
                tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
            B, C, H, W = tensor.shape
            return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

        if len(image_feats_list) > 1 and fit_pca is None:
            target_size = image_feats_list[0].shape[2]
        else:
            target_size = None

        flattened_feats = []
        for feats in image_feats_list:
            flattened_feats.append(flatten(feats, target_size))
        x = torch.cat(flattened_feats, dim=0)

        # Subsample the data if max_samples is set and the number of samples exceeds max_samples
        if max_samples is not None and x.shape[0] > max_samples:
            indices = torch.randperm(x.shape[0])[:max_samples]
            x = x[indices]

        if fit_pca is None:
            if use_torch_pca:
                fit_pca = TorchPCA(n_components=dim).fit(x)
            else:
                from sklearn.decomposition import PCA
                fit_pca = PCA(n_components=dim).fit(x)

        reduced_feats = []
        for feats in image_feats_list:
            x_red = fit_pca.transform(flatten(feats))
            if isinstance(x_red, np.ndarray):
                x_red = torch.from_numpy(x_red)
            x_red -= x_red.min(dim=0, keepdim=True).values
            x_red /= x_red.max(dim=0, keepdim=True).values
            B, C, H, W = feats.shape
            reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

        return reduced_feats, fit_pca

    pca_ret = pca([image_features_bchw])
    camera_view_1, camera_view_2 = pca_ret[0][0][0], pca_ret[0][0][1]
    camera_view_1 = camera_view_1.permute(1, 2, 0).cpu().numpy()
    plt.imshow(camera_view_1)
    plt.show()
