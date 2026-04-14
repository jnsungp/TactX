import dataclasses
import enum
import logging
import socket
import pickle

import tyro
import websocket_policy_server
import yaml
import torch
# from tactile_policy.openpi.models.pi0 import Pi0Config
from tactile_policy.modeling.utils import make_visual_encoder

POLICY_PATH = '/media/ExtHDD/anya/tactile_proj/checkpoints/'
CONFIG_PATH = '/home/anya/Workspace/tactile_benchmark/tactile_policy/configs/models/'

SENSOR_CONFIG_PATH = '/home/anya/Workspace/tactile_benchmark/tactile_policy/configs/sensors/'

# POLICY_PATH = '/home/anyaz/Workspace/tactile/tactile_benchmark/checkpoints/'
# CONFIG_PATH = '/home/anyaz/Workspace/tactile/tactile_benchmark/tactile_policy/configs/models/'

# SENSOR_CONFIG_PATH = '/home/anyaz/Workspace/tactile/tactile_benchmark/tactile_policy/configs/sensors/'


### NOTE: to make the policy server work, you need transform the input to numpy array, and the output is also numpy array.
### The visual_preprocessor is combined into the policy wrapper, so you don't need to preprocess the image before sending it to the server.
def load_policy(policy_path, policy_config_path, device):
    with open(policy_config_path, "r") as fp:
        policy_config = yaml.safe_load(fp)
    policy_type = policy_config["common"]["policy_class"]

    # Load sensor config to determine tactile modality
    tactile_modality = 'none'
    tactile_dim = 0
    if 'sensor_config' in policy_config['common'] and policy_config['common']['sensor_config']:
        sensor_config_path = SENSOR_CONFIG_PATH + policy_config['common']['sensor_config'] + '.yaml'
        with open(sensor_config_path, "r") as fp:
            sensor_config = yaml.safe_load(fp)
        tactile_modality = sensor_config.get('modality', 'none')
        tactile_dim = sensor_config.get('tactile_dim', 0)

    tactile_names = policy_config['common'].get('tactile_names', [])
    num_tactile_sensors = len(tactile_names)

    if policy_type == "ACT":
        policy = torch.jit.load(policy_path, map_location=device).eval().to(device)
        visual_encoder, visual_preprocessor = make_visual_encoder(policy_type, None)

        # NOTE(roger, 2025-04-03): jit traced ACT no longer takes cond_dict due to string conflict
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy, tactile_modality, num_tactile_sensors, tactile_dim):
                super().__init__()
                self.policy = policy
                self.tactile_modality = tactile_modality
                self.num_tactile_sensors = num_tactile_sensors
                self.tactile_dim = tactile_dim

            @torch.no_grad()
            def forward(self, image, qpos, cond_dict, tactile_images=None, tactile_arrays=None):
                image = visual_preprocessor(image.copy()).cuda()
                qpos = torch.from_numpy(qpos).float().cuda()

                if self.tactile_modality == 'image' and tactile_images is not None:
                    tactile_images = visual_preprocessor(tactile_images.copy()).cuda()
                    actions = self.policy(image, qpos, tactile_images)
                elif self.tactile_modality in ['array', 'state'] and tactile_arrays is not None:
                    tactile_arrays = torch.from_numpy(tactile_arrays).float().cuda()
                    actions = self.policy(image, qpos, tactile_arrays)
                else:
                    actions = self.policy(image, qpos)

                actions = {
                    "action": actions.cpu().numpy(),
                }
                return actions

        my_policy_wrapper = polciy_wrapper(policy, tactile_modality, num_tactile_sensors, tactile_dim)
        my_policy_wrapper.eval().to(device)

        return my_policy_wrapper, tactile_modality, num_tactile_sensors, tactile_dim
    else:
        raise ValueError("This only serves ACT policy. Invalid policy type: {}".format(policy_type))

def Warmup_policy(policy, policy_config, tactile_modality, num_tactile_sensors, tactile_dim):
    import numpy as np
    from tqdm import tqdm

    # Get dimensions from config
    state_dim = policy_config['common']['state_dim']
    num_cameras = len(policy_config['common']['camera_names'])

    obs = {
        "image": np.random.randint(0, 255, (1, num_cameras, 3, 240, 320), dtype=np.uint8), # [0, 255]
        "qpos": np.random.randn(1, state_dim).astype(np.float32),
        'cond_dict': {
            "plain_text": ["This is a test prompt for the policy server."]
        },
        "tactile_images": None,
        "tactile_arrays": None,
    }

    # Add tactile inputs based on modality
    if tactile_modality == 'image' and num_tactile_sensors > 0:
        obs["tactile_images"] = np.random.randint(0, 255, (1, num_tactile_sensors, 3, 240, 320), dtype=np.uint8)
    elif tactile_modality in ['array', 'state'] and num_tactile_sensors > 0 and tactile_dim > 0:
        obs["tactile_arrays"] = np.random.randn(1, num_tactile_sensors, tactile_dim).astype(np.float32)

    for i in tqdm(range(10), desc="Warming up policy"):
        _ = policy(obs["image"], obs["qpos"], obs["cond_dict"], obs["tactile_images"], obs["tactile_arrays"])


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    policy_config: str | None = None
    policy_name: str | None = None
    device: str = "cuda"

def main(args: Args) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    policy_path = POLICY_PATH+args.policy_name+"_ckpt/policy_jit.pt"
    policy_config_path = CONFIG_PATH + args.policy_config + '.yaml'
    norm_stats_path = POLICY_PATH + args.policy_name + "_ckpt/dataset_stats.pkl"
    import yaml
    with open(policy_config_path, "r") as fp:
        policy_config = yaml.safe_load(fp)

    with open(norm_stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    policy, tactile_modality, num_tactile_sensors, tactile_dim = load_policy(policy_path, policy_config_path, args.device)
    logging.info(f"Tactile modality: {tactile_modality}, num_sensors: {num_tactile_sensors}, dim: {tactile_dim}")
    Warmup_policy(policy, policy_config, tactile_modality, num_tactile_sensors, tactile_dim)
    logging.info("Policy loaded successfully.")
    embodiment_name = list(norm_stats.keys())[0] 
    logging.info(f"Using normalization stats for embodiment: {embodiment_name}")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=8000,
        metadata={
            'tactile_modality': tactile_modality,
            'tactile_dim': tactile_dim,
            'num_tactile_sensors': num_tactile_sensors,
        },
        norm_stats=norm_stats[embodiment_name],
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))