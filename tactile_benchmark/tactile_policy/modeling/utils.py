import os
import torch
import torch.nn as nn
import torchvision.transforms
from PIL import Image
from typing import Dict, Callable, List

from termcolor import cprint
import pdb
import sys

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def print_params(model):
    """
    Print the number of parameters in each part of the model.
    """    
    params_dict = {}

    all_num_param = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        part_name = name.split('.')[0]
        if part_name not in params_dict:
            params_dict[part_name] = 0
        params_dict[part_name] += param.numel()

    cprint(f'----------------------------------', 'cyan')
    cprint(f'Class name: {model.__class__.__name__}', 'cyan')
    cprint(f'  Number of parameters: {all_num_param / 1e6:.4f}M', 'cyan')
    for part_name, num_params in params_dict.items():
        cprint(f'   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})', 'cyan')
    cprint(f'----------------------------------', 'cyan')


def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif isinstance(value, list):
            # Apply func only to items that have the 'to' method
            result[key] = [func(item) if hasattr(item, 'to') else item for item in value]
        else:
            result[key] = func(value)
    return result

# From WildLMA
class MaskclipBackbone(nn.Module):
    def __init__(self, model_name="ViT-B/16"):
        super().__init__()
        import maskclip_onnx
        self.model_name = model_name
        self.model, self.preprocess = maskclip_onnx.clip.load(
            model_name,
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.patch_size = self.model.visual.patch_size
        self.num_channels = 512
    
    @property
    def num_patches(self):
        # TODO(roger): compute this automatically
        # (336, 448) -> 588
        # (240, 320) -> 300
        # (480, 640) -> 1200
        return 300
    
    @property
    def hidden_size(self):
        if self.model_name == "ViT-B/16":
            return 512
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

    def forward(self, img):
        with torch.no_grad():
            b, _, input_size_h, input_size_w = img.shape
            patch_h = input_size_h // self.patch_size
            patch_w = input_size_w // self.patch_size
            features = self.model.get_patch_encodings(img).to(torch.float32)  # (B, num_patches, 512)
            # clip_aligned_feats_bchw = features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)
            return features

def unit_normalization(image_nchw):
    # Input: NCHW images. N is the number of cameras. np.uint8, [0, 255].
    # Output: processed NCHW that can be used for inference
    image_nchw = torch.from_numpy(image_nchw).float()
    return image_nchw / 255

NORMALIZER = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

def imagenet_normalization(image_nchw):
    # Input: NCHW images. N is the number of cameras. np.uint8, [0, 255].
    # Output: processed NCHW that can be used for inference
    image_nchw = torch.from_numpy(image_nchw).float() / 255
    return NORMALIZER(image_nchw)

def no_normalization(image_nchw):
    return torch.from_numpy(image_nchw)

def make_visual_encoder(policy_class, policy_config):
    if policy_class in ['ACT', 'PI0', '3d_flow', 'dp3']:
        # ACT from OpenTV / PI0 builds its own visual encoder
        # 3D models do not need a visual encoder
        visual_encoder = None
        visual_preprocessor = unit_normalization
        
    elif policy_class == 'GR00T':
        # GR00T builds its own visual encoder
        visual_encoder = None
        visual_preprocessor = no_normalization
    
    elif policy_config["visual_backbone"] == "MASKCLIP":
        # TODO(roger): load different backbones from policy config
        visual_encoder = MaskclipBackbone()
        visual_preprocessor = imagenet_normalization
        return visual_encoder, visual_preprocessor
    elif policy_config["visual_backbone"] == "SIGLIP":
        from .modeling_siglip import SiglipVisionTower
        # Create visual encoder
        SIGLIP_PATH = "google/siglip-so400m-patch14-384"

        visual_encoder = SiglipVisionTower(SIGLIP_PATH, args=None)

        vision_int_mean = tuple(int(x*255) for x in visual_encoder.image_processor.image_mean)

        def visual_preprocessor(image_nchw):
            image_nhwc = image_nchw.transpose((0, 2, 3, 1))
            image_tensor_list = []
            for camera_idx in range(image_nhwc.shape[0]):
                # convert image to a list of PIL images
                img_hwc = image_nhwc[camera_idx]
                image_pil = Image.fromarray(img_hwc)
                image_pil = visual_encoder.expand2square(image_pil, vision_int_mean)
                image_tensor_single = visual_encoder.image_processor.preprocess(image_pil, return_tensors='pt')['pixel_values'][0]
                image_tensor_list.append(image_tensor_single)

            return torch.stack(image_tensor_list, dim=0)
    else:
        raise ValueError(f"Unknown policy class {policy_class}")
    
    return visual_encoder, visual_preprocessor
