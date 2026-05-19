# from joint_embedding_learning.simclr.training import train as simclr_train
# from joint_embedding_learning.barlow_twins.training import train as barlow_twins_train
from joint_embedding_learning.barlow_twins.barlow import BarlowTwins
from joint_embedding_learning.cross_modal.model import (
    CrossModalCLR, CrossModalVAE, CrossModalVAE3, CrossModalCLR3,
    MagneticEncoder, FlexiTacEncoder,
)
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from t3.models import T3
from omegaconf import OmegaConf  # Import OmegaConf

import torch
import os


# def get_trainer(model_name):
#     if model_name == 'simclr':
#         return simclr_train
#     if model_name == 'barlow_twins':
#         return barlow_twins_train
#     else:
#         raise ValueError(f"Model {model_name} not found")

def get_model(model_name, model_details, pretrained=False):
    if model_name == 'simclr':
        encoder = get_resnet(model_details['resnet'], pretrained=pretrained)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        model = SimCLR(encoder, model_details['projection_dim'], n_features)
        return model
    
    elif model_name == 'barlow_twins':
        model = BarlowTwins(model_details['projector'], pretrained=pretrained)
        return model

    elif model_name == 'cross_modal':
        # Image encoder (Daimon) - ResNet backbone
        image_encoder = get_resnet(model_details['resnet'], pretrained=pretrained)
        n_features = image_encoder.fc.in_features  # 512 for ResNet-18, 2048 for ResNet-50

        # Magnetic encoder (eFlesh) - MLP backbone
        magnetic_encoder = MagneticEncoder(
            input_dim=model_details['magnetic_input_dim'],
            hidden_dims=model_details['magnetic_hidden_dims'],
            output_dim=n_features  # match image encoder output dim
        )

        model = CrossModalCLR(
            image_encoder, magnetic_encoder,
            projection_dim=model_details['projection_dim'],
            n_features=n_features
        )
        return model

    elif model_name == 'cross_modal_vae':
        # Image encoder (Daimon) - ResNet backbone
        image_encoder = get_resnet(model_details['resnet'], pretrained=pretrained)
        n_features = image_encoder.fc.in_features

        # Magnetic encoder (eFlesh) - MLP backbone
        magnetic_encoder = MagneticEncoder(
            input_dim=model_details['magnetic_input_dim'],
            hidden_dims=model_details['magnetic_hidden_dims'],
            output_dim=n_features,
        )

        model = CrossModalVAE(
            image_encoder, magnetic_encoder,
            projection_dim=model_details['projection_dim'],
            n_features=n_features,
            image_out_channels=model_details.get('image_out_channels', 3),
            image_out_size=model_details.get('image_out_size', 224),
            image_decoder_channels=model_details.get('image_decoder_channels', [256, 128, 64, 32]),
            magnetic_decoder_hidden=model_details.get('magnetic_decoder_hidden', [256, 128, 64]),
            magnetic_out_dim=model_details.get('magnetic_input_dim', 15),
        )
        return model

    elif model_name == 'cross_modal_vae3':
        image_encoder = get_resnet(model_details['resnet'], pretrained=pretrained)
        n_features = image_encoder.fc.in_features

        magnetic_encoder = MagneticEncoder(
            input_dim=model_details['magnetic_input_dim'],
            hidden_dims=model_details['magnetic_hidden_dims'],
            output_dim=n_features,
        )

        flexi_h = model_details.get('flexitac_in_h', 12)
        flexi_w = model_details.get('flexitac_in_w', 16)
        flexi_enc_channels = model_details.get('flexitac_encoder_channels', [32, 64])
        # Pool to half resolution by default for the spatial bottleneck.
        flexi_pool_h = model_details.get('flexitac_pool_h', max(1, flexi_h // 2))
        flexi_pool_w = model_details.get('flexitac_pool_w', max(1, flexi_w // 2))
        flexitac_encoder = FlexiTacEncoder(
            in_h=flexi_h, in_w=flexi_w,
            channels=tuple(flexi_enc_channels),
            pool_h=flexi_pool_h, pool_w=flexi_pool_w,
            output_dim=n_features,
        )

        model = CrossModalVAE3(
            image_encoder, magnetic_encoder, flexitac_encoder,
            projection_dim=model_details['projection_dim'],
            n_features=n_features,
            image_out_channels=model_details.get('image_out_channels', 3),
            image_out_size=model_details.get('image_out_size', 224),
            image_decoder_channels=model_details.get('image_decoder_channels', [256, 128, 64, 32]),
            magnetic_decoder_hidden=model_details.get('magnetic_decoder_hidden', [256, 128, 64]),
            magnetic_out_dim=model_details.get('magnetic_input_dim', 15),
            flexitac_in_h=flexi_h, flexitac_in_w=flexi_w,
            flexitac_decoder_channels=model_details.get('flexitac_decoder_channels', [64, 32]),
            flexitac_out_channels=model_details.get('flexitac_out_channels', 1),
        )
        return model

    elif model_name == 'cross_modal_clr3':
        # Tri-modal plain-contrastive baseline: same three encoders as
        # cross_modal_vae3 but plain projection heads (projection_dim output,
        # no (mu, logvar)) and no decoders.
        image_encoder = get_resnet(model_details['resnet'], pretrained=pretrained)
        n_features = image_encoder.fc.in_features

        magnetic_encoder = MagneticEncoder(
            input_dim=model_details['magnetic_input_dim'],
            hidden_dims=model_details['magnetic_hidden_dims'],
            output_dim=n_features,
        )

        flexi_h = model_details.get('flexitac_in_h', 12)
        flexi_w = model_details.get('flexitac_in_w', 16)
        flexi_enc_channels = model_details.get('flexitac_encoder_channels', [32, 64])
        flexi_pool_h = model_details.get('flexitac_pool_h', max(1, flexi_h // 2))
        flexi_pool_w = model_details.get('flexitac_pool_w', max(1, flexi_w // 2))
        flexitac_encoder = FlexiTacEncoder(
            in_h=flexi_h, in_w=flexi_w,
            channels=tuple(flexi_enc_channels),
            pool_h=flexi_pool_h, pool_w=flexi_pool_w,
            output_dim=n_features,
        )

        model = CrossModalCLR3(
            image_encoder, magnetic_encoder, flexitac_encoder,
            projection_dim=model_details['projection_dim'],
            n_features=n_features,
        )
        return model

    else:
        raise ValueError(f"Model {model_name} not found")

def get_trained_model(model_name, model_details, dataset_type, run_name, device):
    if model_name == 'simclr':
        
        if run_name == 'pretrained':
            model = get_model(model_name, model_details, pretrained=True)
        
        elif run_name == 'random_init':
            model = get_model(model_name, model_details, pretrained=False)
        
        else:
            model = get_model(model_name, model_details)
            model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_name, run_name, 'model.pt')
            model.load_state_dict(torch.load(model_fp, map_location=device))
        
        model = model.to(device)
        model.eval()

        return model
    
    elif model_name == 'barlow_twins':

        if run_name == 'pretrained':
            model = get_model(model_name, model_details, pretrained=True)

        elif run_name == 'random_init':
            model = get_model(model_name, model_details, pretrained=False)
        
        else:
            model = get_model(model_name, model_details)
            model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_name, run_name, 'model.pt')
            model.load_state_dict(torch.load(model_fp, map_location=device))
            
        model = model.to(device)
        model.eval()

        return model
    
    elif model_name == 'cross_modal':
        if run_name == 'pretrained':
            model = get_model(model_name, model_details, pretrained=True)
        elif run_name == 'random_init':
            model = get_model(model_name, model_details, pretrained=False)
        else:
            model = get_model(model_name, model_details)
            model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_name, run_name, 'model.pt')
            model.load_state_dict(torch.load(model_fp, map_location=device))

        model = model.to(device)
        model.eval()
        return model

    elif model_name == 'cross_modal_vae':
        if run_name == 'pretrained':
            model = get_model(model_name, model_details, pretrained=True)
        elif run_name == 'random_init':
            model = get_model(model_name, model_details, pretrained=False)
        else:
            model = get_model(model_name, model_details)
            model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_name, run_name, 'model.pt')
            model.load_state_dict(torch.load(model_fp, map_location=device))

        model = model.to(device)
        model.eval()
        return model

    elif model_name == 'cross_modal_vae3':
        if run_name == 'pretrained':
            model = get_model(model_name, model_details, pretrained=True)
        elif run_name == 'random_init':
            model = get_model(model_name, model_details, pretrained=False)
        else:
            model = get_model(model_name, model_details)
            model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_name, run_name, 'model.pt')
            model.load_state_dict(torch.load(model_fp, map_location=device))

        model = model.to(device)
        model.eval()
        return model

    elif model_name == 'cross_modal_clr3':
        if run_name == 'pretrained':
            model = get_model(model_name, model_details, pretrained=True)
        elif run_name == 'random_init':
            model = get_model(model_name, model_details, pretrained=False)
        else:
            model = get_model(model_name, model_details)
            model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_name, run_name, 'model.pt')
            model.load_state_dict(torch.load(model_fp, map_location=device))

        model = model.to(device)
        model.eval()
        return model

    elif model_name == 'T3':
        if isinstance(model_details, dict):
            model_details = OmegaConf.create(model_details)
        model = T3(model_details)
        model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_name, run_name)
        model.load_components(model_fp)
        model = model.to(device)
        model.freeze_encoder()
        model.freeze_trunk()
        model.eval()

        return model
        
    else:
        raise ValueError(f"Model {model_name} not found")