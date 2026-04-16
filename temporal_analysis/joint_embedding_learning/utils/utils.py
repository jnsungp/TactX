import os
import torch
import yaml
from simclr.modules import NT_Xent
from joint_embedding_learning.utils.supcon import SupConCrossModalLoss


def load_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    return optimizer, scheduler


def get_loss(args):
    if getattr(args, 'use_supcon', False):
        return SupConCrossModalLoss(temperature=args.temperature)
    return NT_Xent(args.batch_size, args.temperature, 1)


def save_model(model, dataset_name, dataset_type, model_type, dataset_details, model_details, run_name=""):
    path_out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_type, dataset_name + "_run_" + run_name)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    torch.save(model.state_dict(), os.path.join(path_out, "model.pt"))

    dataset_details_path = os.path.join(path_out, "dataset_details.yaml")
    with open(dataset_details_path, 'w') as file:
        yaml.dump(dataset_details, file)

    model_details_path = os.path.join(path_out, "model_details.yaml")
    with open(model_details_path, 'w') as file:
        yaml.dump(model_details, file)


def get_config_details(config_path, config_name):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config[config_name]
