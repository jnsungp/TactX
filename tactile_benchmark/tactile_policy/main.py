import re
import torch
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from tqdm import tqdm
import wandb
import time
import copy

from omegaconf import OmegaConf
import accelerate
from accelerate import Accelerator
from tactile_policy.data_utils_hdt import load_data # data functions
from tactile_policy.data_utils_hdt import compute_dict_mean, set_seed, detach_dict # helper functions
from tactile_policy.modeling.utils import make_visual_encoder
# from tactile_policy.s3_checkpointing_utils import S3WeightSyncManager
# from diffusers.training_utils import EMAModel

# Config file paths
DATASET_CONFIG_DIR = "tactile_policy/configs/datasets"
MODEL_CONFIG_DIR = "tactile_policy/configs/models"
DATA_CONFIG_DIR = "tactile_policy/configs"

def make_policy(policy_class, policy_config, visual_encoder, USE_PRETRAINED=True):
    if policy_class == 'ACT':
        from tactile_policy.modeling.modeling_act import ACTPolicy
        policy = ACTPolicy(policy_config)
    elif policy_class == 'PI0':
        from tactile_policy.modeling.modeling_pi0 import PI0Policy
        policy = PI0Policy(policy_config)
        if USE_PRETRAINED:
            WEIGHT = './pretrained_weights/pi0.safetensors'
            import safetensors.torch
            DEVICE = next(policy.parameters()).device
            loaded = safetensors.torch.load_file(WEIGHT)
            REMOVING_KEYS = ["model.state_proj.weight",
                             "model.state_proj.bias",
                             "model.action_in_proj.weight",
                             "model.action_in_proj.bias",
                             "model.action_out_proj.weight",
                             "model.action_out_proj.bias"]
            state_dict = {k: v.to(DEVICE) for k, v in loaded.items() if k not in REMOVING_KEYS}
            policy.load_state_dict(state_dict, strict=False)  # type: ignore
            print(f"Loaded pretrained weights from {WEIGHT}")
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    # LR will be overwritten by scheduler
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'PI0':
        parameter_list = []
        for name, param in policy.named_parameters():
            VISION_PART = name.startswith('model.paligemma_with_expert.paligemma.vision_tower')
            LANGUAGE_PART = name.startswith('model.paligemma_with_expert.paligemma.language_model')
            if policy.config['model']['train_expert_only']:
                if VISION_PART or LANGUAGE_PART:
                    continue
            if policy.config['model']['freeze_vision_encoder']:
                if VISION_PART:
                    continue
            parameter_list.append(param)
        optimizer = torch.optim.AdamW(
            parameter_list,
            betas=(0.9, 0.95), eps=1e-8,
            weight_decay=1e-10,
            lr=1e-4,
        )
    else:
        raise NotImplementedError
    return optimizer

def main(args, base_dir):
    set_seed(1)
    model_config = OmegaConf.load(args["model_cfg"])
    data_config = OmegaConf.load(args["data_cfg"])
    trainer_config = OmegaConf.merge(model_config, data_config)

    # Load sensor config if specified
    if 'sensor_config' in trainer_config['common'] and trainer_config['common']['sensor_config']:
        sensor_config_path = os.path.join(
            os.path.dirname(args["model_cfg"]),
            '../sensors',
            f"{trainer_config['common']['sensor_config']}.yaml"
        )
        sensor_config = OmegaConf.load(sensor_config_path)
        trainer_config['sensor'] = sensor_config

    # Fall back to YAML chunk_size when not supplied on the command line.
    if args['chunk_size'] is None:
        args['chunk_size'] = int(trainer_config['model']['chunk_size'])

    policy_class = trainer_config["common"]["policy_class"]
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    task_name = "hdt"
    ckpt_dir = "checkpoints/" + args["exptid"] + "_ckpt"

    camera_names = trainer_config['common']['camera_names']
    tactile_names = trainer_config['common']['tactile_names']

    # Load dimensions from config instead of hardcoding
    state_dim = trainer_config['common']['state_dim']
    action_dim = trainer_config['common']['action_dim']
    if policy_class == 'ACT':
        # Build policy config from model config and sensor config
        policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'kl_weight': trainer_config['model']['kl_weight'],
            'hidden_dim': trainer_config['model']['hidden_dim'],
            'chunk_size': args['chunk_size'],
            'dim_feedforward': trainer_config['model']['dim_feedforward'],
            'lr_backbone': float(trainer_config['model']['lr_backbone']),
            'backbone': trainer_config['model']['backbone'],
            'enc_layers': trainer_config['model']['enc_layers'],
            'dec_layers': trainer_config['model']['dec_layers'],
            'nheads': trainer_config['model']['nheads'],
            'camera_names': camera_names,
            'tactile_names': tactile_names,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'image_feature_strategy': trainer_config['model']['image_feature_strategy'],
            'use_language_conditioning': trainer_config['model']['use_language_conditioning'],
            'gradient_accumulation_steps': args['gradient_accumulation_steps'],
        }

        # Add sensor-specific config if available
        if 'sensor' in trainer_config:
            tactile_modality = trainer_config['sensor'].get('modality', 'image')
            policy_config['tactile_modality'] = tactile_modality
            policy_config['tactile_dim'] = trainer_config['sensor']['tactile_dim']
            policy_config['tactile_backbone'] = trainer_config['sensor']['tactile_backbone']
            policy_config['tactile_image_feature_strategy'] = trainer_config['sensor']['tactile_image_feature_strategy']
            policy_config['lr_tactile_backbone'] = float(trainer_config['sensor']['lr_tactile_backbone'])
            if 'tactile_encoding' in trainer_config['sensor']:
                policy_config['tactile_encoding'] = trainer_config['sensor']['tactile_encoding']
            if 'tactile_hidden_layers' in trainer_config['sensor']:
                policy_config['tactile_hidden_layers'] = trainer_config['sensor']['tactile_hidden_layers']
        else:
            tactile_modality = 'none'
            policy_config['tactile_modality'] = tactile_modality
            policy_config['tactile_dim'] = trainer_config['common'].get('tactile_dim', 0)
            policy_config['tactile_backbone'] = trainer_config['model'].get('tactile_backbone', 'resnet18')
            policy_config['tactile_image_feature_strategy'] = trainer_config['model'].get('tactile_image_feature_strategy', 'ACT_linear')
            policy_config['lr_tactile_backbone'] = float(trainer_config['model'].get('lr_tactile_backbone', 1e-4))
    elif policy_class == 'RDT':
        assert "visual_backbone" not in trainer_config
        trainer_config["visual_backbone"] = trainer_config["model"]["backbone"]
        policy_config = trainer_config
    elif policy_class in ['PI0', '3d_flow', 'dp3']:
        # Just use the policy config from the yaml file
        policy_config = trainer_config
    else:
        raise NotImplementedError
    
    from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=ckpt_dir, gradient_accumulation_steps=args['gradient_accumulation_steps'], kwargs_handlers=[kwargs])

    #!
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        # 'episode_len': episode_len,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'seed': args['seed'],
        'camera_names': camera_names,
        'val_and_jit_trace': args['val_and_jit_trace'],
        'task_name': task_name,
        'exptid': args['exptid'],
        'load_pretrained_path': args['load_pretrained_path'],
        'enable_s3_auto_sync': args['enable_s3_auto_sync'],
        'gradient_accumulation_steps': args['gradient_accumulation_steps'],
        'save_ema_model': args['save_ema_model'],
    }
    mode = "disabled" if args["no_wandb"] or args["val_and_jit_trace"] else "online"
    if accelerator.is_main_process:
        wandb.init(project="tactile_policy", name=args['exptid'], group="",
                   entity=None, mode=mode, dir="../data/logs",
                   id=args['exptid'], resume="allow")
        wandb.config.update(config, allow_val_change=True)
    
    visual_encoder, visual_preprocessor = make_visual_encoder(policy_class, policy_config)
    policy = make_policy(policy_class, policy_config, visual_encoder)
    optimizer = make_optimizer(policy_class, policy)

    train_dataloader, val_dataloader, stats = load_data(base_dir,
                                                        trainer_config["data"],
                                                        args["dataset_json"],
                                                        camera_names,
                                                        args["chunk_size"],
                                                        batch_size_train,
                                                        batch_size_val,
                                                        visual_preprocessor,
                                                        tactile_names=tactile_names,
                                                        tactile_modality=tactile_modality)

    if hasattr(policy, 'model') and hasattr(policy.model, 'tactile_image_encoding'):
        if policy.model.tactile_image_encoding in ['pca', 'pca_mlp']:
            if args.get('pca_path') and os.path.exists(args['pca_path']):
                with open(args['pca_path'], 'rb') as f:
                    pca_transformers = pickle.load(f)
                policy.model.tactile_pca_transformers = pca_transformers
                stats['pca_transformers'] = pca_transformers
                if accelerator.is_main_process:
                    print(f"Loaded precomputed PCA from {args['pca_path']}")
            else:
                import h5py
                tactile_images_dict = {name: [] for name in tactile_names}

                for hdf_path in train_dataloader.dataset.dataset_paths:
                    with h5py.File(hdf_path, 'r') as root:
                        for tactile_name in tactile_names:
                            if f'observation.image.{tactile_name}' in root:
                                images = root[f'observation.image.{tactile_name}'][()]
                                if len(images.shape) == 4 and images.shape[-1] == 3:
                                    images = images.transpose(0, 3, 1, 2)
                                tactile_images_dict[tactile_name].append(images)

                for tactile_name in tactile_names:
                    if tactile_images_dict[tactile_name]:
                        tactile_images_dict[tactile_name] = np.concatenate(tactile_images_dict[tactile_name], axis=0)

                policy.model.fit_pca_on_tactile_images(tactile_images_dict)
                stats['pca_transformers'] = policy.model.tactile_pca_transformers

    if accelerator.is_main_process:
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

    train_fn(accelerator, train_dataloader, val_dataloader, policy, optimizer, config)

def maybe_to_tensor(element, to_target):
    if isinstance(element, torch.Tensor):
        return element.to(to_target)
    elif isinstance(element, dict):
        for k, v in element.items():
            if isinstance(v, torch.Tensor):
                element[k] = v.to(to_target)
        return element

#!!! we also change it to tensor in the forward pass
def forward_pass(data, policy):
    device = next(policy.parameters()).device
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = maybe_to_tensor(v, device)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = maybe_to_tensor(data[i], device)
    
    return policy(*(data))

class WarmupMultiplicativeLR(torch.optim.lr_scheduler._LRScheduler):
    """Custom scheduler that multiplies lr by 10 every 1000 steps until reaching max_lr"""
    def __init__(self, optimizer, initial_lr=1e-7, max_lr=1e-4, warmup_period=1000, last_epoch=-1):
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.warmup_period = warmup_period
        # Set initial lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Calculate current multiplier based on epoch
        multiplier = 10 ** (self.last_epoch // self.warmup_period)
        new_lr = min(self.initial_lr * multiplier, self.max_lr)
        return [new_lr for _ in self.base_lrs]

def maybe_load_ckpt(ckpt_dir, seed, train_from_iter):
    ckpt_names = os.listdir(ckpt_dir)
    max_ckpt_name = None
    for ckpt_name in ckpt_names:
        match = re.search(r'policy_iter_(\d+)_seed_(\d+)', ckpt_name)
        if match:
            loaded_iter = int(match.group(1)) 
            cur_seed = int(match.group(2))
            assert cur_seed == seed, f"seed mismatch: {cur_seed} vs {seed}"
            if loaded_iter > train_from_iter:
                train_from_iter = loaded_iter
                max_ckpt_name = ckpt_name
    return train_from_iter, max_ckpt_name

def train_fn(accelerator, train_dataloader, val_dataloader, policy, optimizer, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']

    state = accelerate.state.AcceleratorState()
    process_idx = state.process_index

    # Use different seeds so that multi-GPU don'g get same data
    set_seed(process_idx * 1000 + seed)

    min_val_loss = np.inf

    if config['load_pretrained_path'] is not None:
        print(f"Loading pretrained model from {config['load_pretrained_path']}")
        state_dict = torch.load(config['load_pretrained_path'], map_location=next(policy.parameters()).device, weights_only=True)
        policy.load_state_dict(state_dict, strict=False)

    # use constant LR scheduler
    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=num_epochs)
    # Cycling cosine
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    total_steps = int(num_epochs * accelerator.num_processes / config['gradient_accumulation_steps'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')

    train_dataloader, policy, optimizer, scheduler = accelerator.prepare(train_dataloader, policy, optimizer, scheduler)

    if config['load_pretrained_path'] is not None:
        print(f"Loading pretrained model from {config['load_pretrained_path']}")
        state_dict = torch.load(config['load_pretrained_path'], map_location=next(policy.parameters()).device, weights_only=True)
        ret = policy.load_state_dict(state_dict, strict=False)

    train_from_iter = 0

    if accelerator.is_main_process and config['enable_s3_auto_sync']:
        s3_sync_manager = S3WeightSyncManager(ckpt_name=ckpt_dir, s3_base_path="weights_auto_sync", bucket_name="cxx-tv")
        s3_sync_manager.start()

        s3_sync_manager.blocking_download_latest_ckpt()
        
    accelerator.wait_for_everyone()

    train_from_iter, max_ckpt_name = maybe_load_ckpt(ckpt_dir, seed, train_from_iter)
    if train_from_iter > 0:
        print(f"Resuming from iter {train_from_iter}")
        accelerator.load_state(os.path.join(ckpt_dir, max_ckpt_name))

    policy.train()
    if config['save_ema_model']:
        ema = EMAModel(policy.parameters(), decay=0.99)
        ema_policy = copy.deepcopy(policy).to('cpu')
    cur_iter = train_from_iter

    with tqdm(total=num_epochs, initial=train_from_iter) as pbar:
        for data in train_dataloader:
            image_data, qpos_data, action_data, is_pad, conditioning_dict, tactile_images, tactile_arrays = data

            train_state = {
                'cur_train_step': cur_iter,
                'max_train_steps': num_epochs * len(train_dataloader),
                'ema_model': None, # TODO(roger): EMA model is not used in flow matching now.
            }

            # TODO(roger): consolidate everything into dictionary
            assert isinstance(conditioning_dict, dict)
            assert 'train_state' not in conditioning_dict
            conditioning_dict['train_state'] = train_state

            if cur_iter >= num_epochs or config['val_and_jit_trace']:
                break

            if cur_iter % 1000 == 0:
                # validation
                with torch.no_grad():
                    policy.eval()
                    validation_dicts = []
                    for batch_idx, val_data in enumerate(val_dataloader):
                        # TODO(roger): consolidate everything into dictionary
                        assert 'train_state' not in val_data[4]
                        val_data[4]['train_state'] = train_state
                        forward_dict = forward_pass(val_data, policy)
                        validation_dicts.append(forward_dict)
                        if batch_idx > 20:
                            break

                    validation_summary = compute_dict_mean(validation_dicts)
                    
                    epoch_val_loss = validation_summary['loss']
                if accelerator.is_main_process:
                    print(f'\n Iter {cur_iter}')
                    for k in list(validation_summary.keys()):
                        validation_summary[f'val/{k}'] = validation_summary.pop(k)     

                    wandb.log(validation_summary, step=cur_iter)
                    print(f'Val loss:   {epoch_val_loss:.5f}')
                    summary_string = ''
                    for k, v in validation_summary.items():
                        try:
                            raw_value = v.item()
                        except AttributeError:
                            raw_value = v
                        summary_string += f'{k}: {raw_value:.3f} '
                    print(summary_string)

                    if config['val_and_jit_trace']:
                        break

                policy.train()

            with accelerator.accumulate(policy):
                forward_dict = policy(*(data))
                # backward
                loss = forward_dict['loss']

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if config['save_ema_model']:
                    ema.step(policy.parameters())

            pbar.update(1)
            cur_iter += 1
            
            if accelerator.is_main_process:
                epoch_summary = detach_dict(forward_dict)
                epoch_summary['lr'] = torch.tensor(scheduler.get_last_lr()[0])
                summary_string = ''
                for k, v in epoch_summary.items():
                    try:
                        raw_value = v.item()
                    except AttributeError:
                        raw_value = v
                    summary_string += f'{k}: {raw_value:.3f} '
                # print(summary_string)
                wandb.log(epoch_summary, step=cur_iter)

                #! save ckpt
                if cur_iter % 10000 == 0 and cur_iter != 0:
                    ckpt_path = os.path.join(ckpt_dir, f'policy_iter_{cur_iter}_seed_{seed}')
                    # This is similar to torch.save, but it also saves optimizer/scheduler states
                    accelerator.save_state(ckpt_path, safe_serialization=False)
                    if config['save_ema_model']:
                        ema.copy_to(ema_policy.parameters())
                        accelerator.save(ema_policy.state_dict(), os.path.join(ckpt_path, f'ema_model.bin'))
                    print(f'Saved ckpt at iter {cur_iter}')
                    if accelerator.is_main_process and config['enable_s3_auto_sync']:
                        s3_sync_manager.add_sync_task(ckpt_dir)

    if config['val_and_jit_trace']:
        # JIT trace
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, image, qpos):
                return self.policy(image, qpos, conditioning_dict=None)

        # TRACING_DEVICE = 'cuda'
        TRACING_DEVICE = 'cuda'
        # Rollout validation
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(TRACING_DEVICE)

        # Benchmark
        image, qpos, _, _, conditioning_dict = data
        image = image.to(TRACING_DEVICE)
        qpos = qpos.to(TRACING_DEVICE)
        # warm up
        for _ in range(10):
            trajectory = my_policy_wrapper(image[0:1], qpos[0:1])
        # benchmark speed
        start = time.time()
        N_iters = 50
        for _ in range(N_iters):
            trajectory = my_policy_wrapper(image[0:1], qpos[0:1])
        end = time.time()
        print("Total time: ", end - start)
        print(f"Rollout speed: {N_iters / (end - start)} Hz")

        # Jit trace
        image_data = torch.rand(image[0:1].shape, device=TRACING_DEVICE)
        qpos_data = torch.rand(qpos[0:1].shape, device=TRACING_DEVICE)
        input_data = (image_data, qpos_data)

        traced_policy = torch.jit.trace(my_policy_wrapper, input_data)

        traced_path = os.path.join(ckpt_dir, f'policy_traced.pt')
        traced_policy.save(traced_path)
        del traced_policy
        torch.cuda.empty_cache()

        loaded_policy = torch.jit.load(traced_path)
        # Manually set seed to make diffusion sampling deterministic in comparisons
        torch.random.manual_seed(0)
        jit_output = loaded_policy(image_data, qpos_data)
        torch.random.manual_seed(0)
        vanilla_output = my_policy_wrapper(image_data, qpos_data)

        l1_err = torch.nn.functional.l1_loss(jit_output, vanilla_output, reduction='none').cpu().detach().numpy()
        assert (l1_err < 1e-3).all(), f"JIT trace error: {l1_err.max()}"

        exit(0)
    
    if accelerator.is_main_process and config['enable_s3_auto_sync']:
        s3_sync_manager.wait_for_completion()
        s3_sync_manager.stop()
    
    accelerator.wait_for_everyone()
            
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False, default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--val_and_jit_trace', action='store_true')
    parser.add_argument('--exptid', action='store', type=str, help='experiment id', required=True)
    parser.add_argument('--epoch', action='store', type=str, help='epoch num', required=False)
    parser.add_argument('--dataset_json', type=str, help='dataset config name (e.g., pnp_fsr_state_left)', required=True)
    parser.add_argument('--model_cfg', type=str, help='model config name (e.g., act_resnet)', required=True)
    parser.add_argument('--data_cfg', type=str, help='data config name (e.g., data_base)', required=False, default='data_base')
    parser.add_argument('--load_pretrained_path', type=str, help='path to load pretrained model', required=False)
    parser.add_argument('--enable_s3_auto_sync', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps', required=False)
    parser.add_argument('--save_ema_model', action='store_true', default=False)
    parser.add_argument('--pca_path', type=str, help='path to precomputed PCA transformers', required=False)
    args = vars(parser.parse_args())

    # Build full config paths from simple names
    args['dataset_json'] = os.path.join(DATASET_CONFIG_DIR, f"{args['dataset_json']}.json")
    args['model_cfg'] = os.path.join(MODEL_CONFIG_DIR, f"{args['model_cfg']}.yaml")
    args['data_cfg'] = os.path.join(DATA_CONFIG_DIR, f"{args['data_cfg']}.yaml")

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/recordings/processed')
    assert os.path.exists(base_dir)

    main(args, base_dir)
