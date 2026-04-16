from joint_embedding_learning.cross_modal.model import CrossModalCLR, MagneticEncoder
from simclr.modules import get_resnet

import torch
import os


def get_model(model_details, pretrained=False):
    image_encoder = get_resnet(model_details['resnet'], pretrained=pretrained)
    n_features = image_encoder.fc.in_features  # 512 for ResNet-18, 2048 for ResNet-50

    magnetic_encoder = MagneticEncoder(
        input_dim=model_details['magnetic_input_dim'],
        hidden_dims=model_details['magnetic_hidden_dims'],
        output_dim=n_features
    )

    model = CrossModalCLR(
        image_encoder, magnetic_encoder,
        projection_dim=model_details['projection_dim'],
        n_features=n_features
    )
    return model


def get_trained_model(model_details, dataset_type, run_name, device):
    if run_name == 'pretrained':
        model = get_model(model_details, pretrained=True)
    elif run_name == 'random_init':
        model = get_model(model_details, pretrained=False)
    else:
        model = get_model(model_details)
        model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, 'cross_modal', run_name, 'model.pt')
        model.load_state_dict(torch.load(model_fp, map_location=device))

    model = model.to(device)
    model.eval()
    return model
