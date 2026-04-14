import dataclasses
import enum
import logging
import socket
import pickle

import tyro
import websocket_policy_server
import yaml
import torch
from tactile_policy.openpi.models.pi0 import Pi0Config
from tactile_policy.modeling.utils import make_visual_encoder

### NOTE: to make the policy server work, you need transform the input to numpy array, and the output is also numpy array.
### The visual_preprocessor is combined into the policy wrapper, so you don't need to preprocess the image before sending it to the server.
def load_policy(policy_path, policy_config_path, device):
    with open(policy_config_path, "r") as fp:
        policy_config = yaml.safe_load(fp)
    policy_type = policy_config["common"]["policy_class"]
    if policy_type == "ACT":
        policy = torch.jit.load(policy_path, map_location=device).eval().to(device)
        visual_encoder, visual_preprocessor = make_visual_encoder(policy_type, None)

        # NOTE(roger, 2025-04-03): jit traced ACT no longer takes cond_dict due to string conflict
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos, cond_dict):
                image = visual_preprocessor(image.copy()).cuda()
                qpos = torch.from_numpy(qpos).float().cuda()
                actions = self.policy(image, qpos)
                actions = {
                    "action": actions.cpu().numpy(),
                }
                return actions
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(device)

        return my_policy_wrapper
    elif policy_type == "RDT":
        visual_encoder, visual_preprocessor = make_visual_encoder(policy_type, {"visual_backbone": "MASKCLIP"})

        from hdt.modeling.modeling_hdt import HumanDiffusionTransformer
        policy = HumanDiffusionTransformer(
            action_dim=policy_config["common"]["state_dim"],
            pred_horizon=policy_config["common"]["action_chunk_size"],
            config=policy_config,
            lang_token_dim=policy_config["model"]["lang_token_dim"],
            img_token_dim=policy_config["model"]["img_token_dim"],
            state_token_dim=policy_config["model"]["state_token_dim"],
            max_lang_cond_len=policy_config["dataset"]["tokenizer_max_length"],
            visual_encoder=visual_encoder,
            lang_pos_embed_config=[
                # Similarly, no initial pos embed for language
                ("lang", -policy_config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=torch.float32,
        )
        checkpoint = torch.load(policy_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(new_state_dict, strict=True)
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos, cond_dict):
                assert 'language_embeddings' in cond_dict
                valid_mask = torch.ones(cond_dict['language_embeddings'].shape[:2], dtype=torch.bool).to(device=device)
                cond_dict['language_embeddings_mask'] = valid_mask
                image = visual_preprocessor(image).cuda()
                qpos = torch.from_numpy(qpos).float().cuda()
                return self.policy(image, qpos, conditioning_dict=cond_dict)
    
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(device)

        return my_policy_wrapper
    elif policy_type == "PI0":
        visual_encoder, visual_preprocessor = make_visual_encoder(policy_type, policy_config)

        from hdt.modeling.modeling_pi0 import PI0Policy
        policy = PI0Policy(policy_config)

        checkpoint = torch.load(policy_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(new_state_dict, strict=True)
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos, cond_dict):
                image = visual_preprocessor(image).cuda()
                qpos = torch.from_numpy(qpos).float().cuda()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return self.policy(image, qpos, conditioning_dict=cond_dict)
    
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(device)

        return my_policy_wrapper, visual_preprocessor
    elif policy_type == "DP":
        visual_encoder, visual_preprocessor = make_visual_encoder(policy_type, {"visual_backbone": "MASKCLIP"})
        
        from hdt.modeling.modeling_vanilla_dp import DiffusionPolicy
        policy = DiffusionPolicy(action_dim=128,
            chunk_size=64,
            img_token_dim=visual_encoder.hidden_size,
            state_token_dim=128,
            num_inference_timesteps=20,
            visual_encoder=visual_encoder)
        checkpoint = torch.load(policy_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(new_state_dict, strict=True)
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos, cond_dict):
                image = visual_preprocessor(image).cuda()
                qpos = torch.from_numpy(qpos).float().cuda()
                return self.policy(image, qpos, conditioning_dict=cond_dict)
    
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().cuda()

        return my_policy_wrapper, visual_preprocessor
    elif policy_type == "3d_flow":
        from hdt.modeling.modeling_flow import Flow_consistency_3d
        policy = Flow_consistency_3d(**policy_config["model"])

        visual_encoder, visual_preprocessor = make_visual_encoder(policy_type, policy_config)

        checkpoint = torch.load(policy_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(new_state_dict, strict=True)
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos, cond_dict):
                image = visual_preprocessor(image).cuda()
                qpos = torch.from_numpy(qpos).float().cuda()
                return self.policy(image, qpos, conditioning_dict=cond_dict)
    
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(device)

        return my_policy_wrapper, visual_preprocessor
    elif policy_type == "GR00T":
        visual_encoder, visual_preprocessor = make_visual_encoder("GR00T", policy_config)

        from hdt.modeling.modeling_gr00t import GR00T_N1, GR00T_N1Config
        config = GR00T_N1Config.from_pretrained("../hdt/pretrained_weights/gr00t.json")
        config.action_dim = policy_config['common']['action_dim']
        config.action_head_cfg['action_dim'] = policy_config['common']['action_dim']
        config.action_head_cfg['action_horizon'] = policy_config['model']['chunk_size']
        config.action_horizon = policy_config['model']['chunk_size']
        config.action_head_cfg['max_action_dim'] = policy_config['common']['action_dim']
        config.action_head_cfg['max_state_dim'] = policy_config['common']['state_dim']
        if policy_config['model']['resize_imgs'] is not None:
            config.resize_imgs = policy_config['model']['resize_imgs']
        policy = GR00T_N1(config=config, local_model_path="")
        policy.compute_dtype = "bfloat16"
        policy.config.compute_dtype = "bfloat16"
        checkpoint = torch.load(policy_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(new_state_dict, strict=True)

        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos, cond_dict):
                image = visual_preprocessor(image).cuda()
                qpos = torch.from_numpy(qpos).float().cuda()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return self.policy(image, qpos, conditioning_dict=cond_dict)
    
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().cuda()

        return my_policy_wrapper, visual_preprocessor     
    elif policy_type == "PI0_JAX":
        from tactile_policy.remote_inference.jax_policy_warpper import JaxPolicyWrapper
        model_config = Pi0Config(action_dim=128, action_horizon=64, max_token_len=48)
        policy = JaxPolicyWrapper(model_config, policy_path)
        return policy
    elif policy_type == "PI0_LORA_JAX":
        from tactile_policy.remote_inference.jax_policy_warpper import JaxPolicyWrapper
        model_config = Pi0Config(action_dim=128, action_horizon=64, max_token_len=48, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
        policy = JaxPolicyWrapper(model_config, policy_path)
        return policy
    else:
        raise ValueError("Invalid policy type: {}".format(policy_type))

def Warmup_policy(policy):
    import numpy as np
    from tqdm import tqdm
    obs = {
        "image": np.random.randint(0, 255, (1, 2, 3, 240, 320), dtype=np.uint8), # [0, 255]
        "qpos": np.random.randn(1, 128).astype(np.float32),
        'cond_dict': {
            "plain_text": ["This is a test prompt for the policy server."]
        }
    }
    for i in tqdm(range(10), desc="Warming up policy"):
        _ = policy(obs["image"], obs["qpos"], obs["cond_dict"])


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    policy_config_path: str | None = None
    policy_path: str | None = None
    norm_stats_path: str | None = None
    device: str = "cuda"

def main(args: Args) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    with open(args.norm_stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    policy = load_policy(args.policy_path, args.policy_config_path, args.device)
    Warmup_policy(policy)
    logging.info("Policy loaded successfully.")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=8000,
        metadata={},
        norm_stats=norm_stats['franka'],
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))