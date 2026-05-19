# tactile_benchmark

ACT policy training and deployment with tactile sensing.

## Installation

### Prerequisites

Install [Miniconda](https://docs.anaconda.com/free/miniconda/) if not already installed.

### Setup

```bash
conda create -y -n tactile_bench python=3.11
conda activate tactile_bench
cd tactile_benchmark
bash create_env.sh
```

### Optional: PI0 Support

```bash
pip install -r requirements_pi0.txt
```

## Data Format

### HDF5 Episode Files

Training expects an hdf5 file (`episode_####.hdf5`) with this struct:

**Required datasets:**
- `observation.state`: [T, 128] float32 - Robot state (padded to 128 dimensions)
- `action`: [T, 128] float32 - Action commands (padded to 128 dimensions)
- `observation.image.cam0`: [T, 240, 320, 3] uint8 - Wrist camera RGB frames
- `observation.image.cam1`: [T, 240, 320, 3] uint8 - Third-person camera RGB frames
- `observation.image.tac0`: [T, 240, 320, 3] uint8 - left tactile sensor image representation (daimon is highest res of 240 x 320)
- `observation.image.tac1`: [T, 240, 320, 3] uint8 - right tactile sensor image representation 
- `observation.tac0`: [T, idk] float32 - flattened tactile array
- `observation.tac1`: [T, idk] float32 - flattened tactile array


**HDF5 attributes:**
- `original_state_dim`: int - Original state dimension before padding
- `original_action_dim`: int - Original action dimension before padding
- `tactile_dim`: int - Tactile sensor dimension (if applicable)

### Directory Structure

Data dir structure needed for training:
```
data/recordings/processed/
├── <dataset_name_1>/
│   ├── episode_0000.hdf5
│   ├── episode_0001.hdf5
│   └── ...
├── <dataset_name_2>/
│   ├── episode_0000.hdf5
│   └── ...
└── <validation_dataset>/
    ├── episode_0000.hdf5
    └── ...
```

### Data Conversion from ACEF
There is a script (tactile_policy/process_acef_data.py) to convert to the above format from each episode being represented as 4 `.mp4` files (2 camera 2 tactile videos) and a `.pkl` file (state, action, timestamps). 

Original structure from ACEF Data collection system: 
```
Input format (per episode):
  episode_####/
    ├── cam0.mp4 (wrist camera)
    ├── cam1.mp4 (3rd person camera)
    ├── tac0.mp4 (tactile sensor 0)
    ├── tac1.mp4 (tactile sensor 1)
    └── low_dim_obs.pkl (contains state, action, and tactile data)
```

Where `low_dim_obs.pkl` has this entry structurte:
```
{'state': array([...]),
  'state_timestamp': t,
  'img_timestamp': [t0, t1],
  'action': array([...]),
  'tactile': array([...]),
  'data_type': True}
```


### Configurations

#### Dataset Config

Create a JSON file in `tactile_policy/configs/datasets/` to specify train/val splits:

```json
{
    "train": ["dataset_name_1", "dataset_name_2"],
    "val": ["validation_dataset"]
}
```

The dataset names correspond to folder names in `data/recordings/processed/`.

#### Sensor Config
For each sensor this is a config file in `/tactile_policy/configs/sensors`, that is refernced by the model config to know which sensor is being used.

Example for FSR:
```
sensor_type: fsr
modality: array ['array' - raw data, 'image'- image of tactile data for visual sensors, 'none' - dont use any tactile data, use this to train vision only]
tactile_dim: 1 [FSR is just one scalar, it is assumed there are always two sensors]

# tactile img config (ignored if mode is not img)
tactile_backbone: resnet18
tactile_image_feature_strategy: ACT_linear

# tactile array config (ignored if mode is not array)
lr_tactile_backbone: 1e-4
tactile_encoding: mlp
tactile_hidden_layers: [64, 128]
```

These need to be filled out for each sensor. Fill out the flattened dimensions in `tactile_dim` for a single sensor (ex: flexitac is an array of 12x32 -> 384, eflesh has normal and sheer force, and 5 points of reading so 20)

#### Model Config
Stored in `/tactile_policy/configs/models`
Example: 

```
common:
  policy_class: ACT
  state_dim: 128
  action_dim: 128
  sensor_config: fsr
  camera_names: ['cam0', 'cam1']
  tactile_names: ['tac0', 'tac1']

model:
  enc_layers: 4
  dec_layers: 7
  nheads: 8
  hidden_dim: 512
  kl_weight: 10
  dim_feedforward: 3200
  lr_backbone: 1e-5
  backbone: resnet18
  image_feature_strategy: ACT_linear


  use_language_conditioning: False
  num_queries: 100
  chunk_size: 100
  ```

## Usage

### Training

Run from `tactile_benchmark/` (paths to configs and data are relative to that directory).

**Cross-modal tactile latents (16-d FlexiTac episodes; HDF5 must already contain latent observations — see repo root `README.md` → *Latent conversion* and `scripts/convert_tac_to_latent.py`):**

```bash
cd tactile_benchmark && \
CUDA_VISIBLE_DEVICES=0 python tactile_policy/main.py \
  --batch_size 8 \
  --num_epochs 50000 \
  --lr 1e-5 \
  --chunk_size 64 \
  --exptid pnp_flexi_cmlatent_mmd \
  --dataset_json pnp_flexi_latent \
  --model_cfg act_cross_modal_pnp_z16
```

**Raw FlexiTac array (MLP on flattened tactile):**

```bash
cd tactile_benchmark && \
CUDA_VISIBLE_DEVICES=0 python tactile_policy/main.py \
  --batch_size 8 \
  --num_epochs 50000 \
  --lr 1e-4 \
  --chunk_size 64 \
  --exptid pnp_flexi_raw_mlp \
  --dataset_json pnp_flexi \
  --model_cfg act_flexi
```

## Worked example: pnp_tennis_ball (Daimon)

End-to-end pipeline: pre-processed HDF5 → policy. Covers three tactile variants (raw, cross-modal latent, vision-only) and the hyperparameter sweep.

### 1. Pre-processing

**1a. Unwrap 2π action offsets.** The gello teleop agent can record action joints as `state + k*2π`. Undo it (writes to a new folder, non-destructive):

```bash
python scripts/unwrap_action_2pi.py \
  --src tactile_benchmark/data/recordings/processed/pnp_tennis_ball \
  --dst tactile_benchmark/data/recordings/processed/pnp_tennis_ball_unwrapped

python scripts/unwrap_action_2pi.py \
  --src tactile_benchmark/data/recordings/processed/pnp_tennis_ball_val \
  --dst tactile_benchmark/data/recordings/processed/pnp_tennis_ball_unwrapped_val
```

Verify every `|a-s|max` is well below π (typically <0.5):
```bash
python -c "
import h5py, numpy as np
with h5py.File('tactile_benchmark/data/recordings/processed/pnp_tennis_ball_unwrapped/episode_0000.hdf5') as f:
    a=f['action'][:, :8]; s=f['observation.state'][:, :8]
for d in range(8):
    print(f'  d{d}: |a-s|max={np.abs(a[:,d]-s[:,d]).max():.3f}')"
```

**1b. (Latent variant only) Generate tactile latents** with the cross-modal CLR encoder:

```bash
python scripts/convert_tac_to_latent.py \
  --checkpoint_dir checkpoints/cross_modal/cross_modal/<encoder_run> \
  --daimon_input_dir  tactile_benchmark/data/recordings/processed/pnp_tennis_ball_unwrapped \
  --daimon_output_dir tactile_benchmark/data/recordings/processed/pnp_tennis_ball_unwrapped_latent

# Same for val split.
```

Note: `convert_tac_to_latent.py` currently does **not** copy root HDF5 attributes. Patch the required attrs after conversion:
```bash
python -c "
import h5py, glob
for d in ['tactile_benchmark/data/recordings/processed/pnp_tennis_ball_unwrapped_latent',
          'tactile_benchmark/data/recordings/processed/pnp_tennis_ball_unwrapped_latent_val']:
    for p in sorted(glob.glob(d+'/episode_*.hdf5')):
        with h5py.File(p,'a') as f:
            f.attrs['embodiment'] = 'franka'
            f.attrs['description'] = 'pnp tennis ball daimon latent'"
```

Also pick a sensor config whose `tactile_dim` matches the encoder's projection. The default [cross_modal_latent.yaml](tactile_policy/configs/sensors/cross_modal_latent.yaml) is 64-d. For a 16-d encoder, use [cross_modal_latent_z16.yaml](tactile_policy/configs/sensors/cross_modal_latent_z16.yaml).

### 2. Configs

Dataset jsons in [tactile_policy/configs/datasets/](tactile_policy/configs/datasets/):
- `pnp_tennis_ball_unwrapped.json` — raw daimon images
- `pnp_tennis_ball_unwrapped_latent.json` — 16-d cross-modal latents

Model yamls in [tactile_policy/configs/models/](tactile_policy/configs/models/):
- `act_daimon_v2.yaml` — raw daimon, `kl_weight: 1`, `lr_backbone: 1e-4`, `chunk_size: 16`
- `act_cross_modal_pnp_z16.yaml` — 16-d latent tactile
- `act_daimon_vis_v2.yaml` — vision only (tactile disabled via sensor modality `"none"`)

These hyperparameters (vs the defaults of `kl_weight: 10`, `lr_backbone: 1e-5`, `chunk_size: 64`) were tuned to fix mode collapse on small Daimon datasets (~10 k frames).

### 3. Training

**Raw daimon:**
```bash
cd tactile_benchmark && \
CUDA_VISIBLE_DEVICES=0 python tactile_policy/main.py \
  --batch_size 8 --num_epochs 50000 --lr 1e-5 \
  --exptid pnp_tennis_ball_raw_cs16 \
  --dataset_json pnp_tennis_ball_unwrapped \
  --model_cfg act_daimon_v2 \
  --chunk_size 16
```

**Cross-modal latent (16-d):**
```bash
cd tactile_benchmark && \
CUDA_VISIBLE_DEVICES=0 python tactile_policy/main.py \
  --batch_size 8 --num_epochs 50000 --lr 1e-5 \
  --exptid pnp_tennis_ball_latent_cs16 \
  --dataset_json pnp_tennis_ball_unwrapped_latent \
  --model_cfg act_cross_modal_pnp_z16 \
  --chunk_size 16
```

**Vision only (no tactile):**
```bash
cd tactile_benchmark && \
CUDA_VISIBLE_DEVICES=0 python tactile_policy/main.py \
  --batch_size 8 --num_epochs 50000 --lr 1e-5 \
  --exptid pnp_tennis_ball_vis_cs16 \
  --dataset_json pnp_tennis_ball_unwrapped \
  --model_cfg act_daimon_vis_v2 \
  --chunk_size 16
```

### 4. Export JIT and offline eval

```bash
# Export (example for raw)
cd tactile_benchmark/tactile_policy/remote_inference && \
python export_act_to_jit.py \
  --policy_name pnp_tennis_ball_raw_cs16 \
  --policy_config act_daimon_v2 && \
cd ../../..

# Offline replay on val: produces per-episode PNG + npz under eval_outputs/
python scripts/jit_offline_inference.py \
  --jit_path   tactile_benchmark/checkpoints/pnp_tennis_ball_raw_cs16_ckpt/policy_jit.pt \
  --stats_path tactile_benchmark/checkpoints/pnp_tennis_ball_raw_cs16_ckpt/dataset_stats.pkl \
  --model_cfg  tactile_benchmark/tactile_policy/configs/models/act_daimon_v2.yaml \
  --hdf_path   tactile_benchmark/data/recordings/processed/pnp_tennis_ball_unwrapped_val \
  --out_dir    tactile_benchmark/checkpoints/pnp_tennis_ball_raw_cs16_ckpt/eval_outputs \
  --real_action_dims 8
```

Note: `export_act_to_jit.py` reads `chunk_size` from the yaml with no CLI override. If you train multiple chunk sizes, make per-chunk-size yaml variants (`act_daimon_v2_cs8.yaml`, `_cs32.yaml`, etc.) — otherwise export will shape-mismatch the checkpoint.

Key offline metrics (printed by `jit_offline_inference.py`):
- **MAE** on first 8 action dims — lower is better.
- **pred_std / GT_std** per dim — near 1.0 means the policy tracks trajectory variation. Near 0 on joint dims (0-6) indicates mode collapse even if MAE looks OK.

### 5. Chunk-size sweep

Two-stage sweep scripts in [bash_scripts/](bash_scripts/):

```bash
# Stage 1: train cs ∈ {8, 16, 32, 64}, cleans intermediate iters
bash tactile_benchmark/bash_scripts/sweep_chunk_size.sh

# Stage 2: export JIT + offline eval for each trained ckpt, prints MAE summary
bash tactile_benchmark/bash_scripts/sweep_chunk_size_eval.sh
```

Override chunk list / epochs via env vars:
```bash
EPOCHS=10000 CHUNK_SIZES="16" bash tactile_benchmark/bash_scripts/sweep_chunk_size.sh
```


## Accelerator

**This has not been tested and is legacy code + instruction**

For more sophisticated training, such as BF16, torch.compile, or multi-GPU training, the training script supports huggingface accelerator.

Start by configuring a `config.yaml` file.

```bash
accelerate config --config_file ./my_accelerator.yaml
```

Then

```bash
accelerate launch --config_file ./my_accelerator.yaml main.py --chunk_size 64 --batch_size 42 --num_epochs 50000 --lr 1e-4 --seed 0 --exptid sim_aloha_wrist --dataset_json sim_test --model_cfg opentv_act
```

To run pi0+PyTorch training, an example command is

```bash
accelerate launch --main_process_port 29501 --config_file ./4_gpu_bf16.yaml  main.py --chunk_size 64 --batch_size 16 --num_epochs 50000 --lr 5e-5 --seed 0 --exptid 'test_job' --dataset_json_path test.json  --model_cfg pi0_all


1. Convert hammer dirs to latent
From ~/cross_modal_tactile, cmt env (sensor D=daimon, E=eflesh, F=flexi):


conda activate cmt
cd ~/cross_modal_tactile
CKPT=checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_static12_run_infonce_static12_efs3_20260517_224612
PROC=tactile_benchmark/data/recordings/processed

python scripts/convert_tac_to_latent_sachin.py --checkpoint_dir $CKPT --checkpoint_file model_ep400.pt --sensor D \
  --input_dir $PROC/dai_dai_hammer       --output_dir $PROC/dai_dai_hammer_latent_static12       --device cuda:0
python scripts/convert_tac_to_latent_sachin.py --checkpoint_dir $CKPT --checkpoint_file model_ep400.pt --sensor E \
  --input_dir $PROC/eflesh_eflesh_hammer --output_dir $PROC/eflesh_eflesh_hammer_latent_static12 --device cuda:0
python scripts/convert_tac_to_latent_sachin.py --checkpoint_dir $CKPT --checkpoint_file model_ep400.pt --sensor F \
  --input_dir $PROC/flexi_flexi_hammer   --output_dir $PROC/flexi_flexi_hammer_latent_static12   --device cuda:0
2. Split 3 episodes into val
Moves the last 3 sorted episodes of each latent dir into a *_val_* dir:


cd ~/cross_modal_tactile/tactile_benchmark/data/recordings/processed
for s in dai_dai eflesh_eflesh flexi_flexi; do
  src=${s}_hammer_latent_static12
  val=${s}_hammer_val_latent_static12
  mkdir -p "$val"
  ls "$src"/episode_*.hdf5 | sort -V | tail -3 | xargs -I{} mv {} "$val"/
  echo "$s -> train $(ls $src/episode_*.hdf5 | wc -l) / val $(ls $val/episode_*.hdf5 | wc -l)"
done
3. Train policy (3 cmds)
From ~/cross_modal_tactile/tactile_benchmark, tactile_bench env, in tmux:


tmux kill-session -t hammer_daimon_latent_static12 2>/dev/null; tmux new-session -d -s hammer_daimon_latent_static12 "source /home/junsung/anaconda3/etc/profile.d/conda.sh && conda activate tactile_bench && cd ~/cross_modal_tactile/tactile_benchmark && PYTHONPATH=/home/junsung/cross_modal_tactile/tactile_benchmark CUDA_VISIBLE_DEVICES=0 python tactile_policy/main.py --batch_size 8 --num_epochs 50000 --lr 1e-5 --exptid hammer_daimon_latent_static12 --dataset_json hammer_daimon_latent_static12 --model_cfg act_cross_modal_pnp_z16 --chunk_size 64 2>&1 | tee train_logs/hammer_daimon_latent_static12.log; exec bash"

tmux kill-session -t hammer_eflesh_latent_static12 2>/dev/null; tmux new-session -d -s hammer_eflesh_latent_static12 "source /home/junsung/anaconda3/etc/profile.d/conda.sh && conda activate tactile_bench && cd ~/cross_modal_tactile/tactile_benchmark && PYTHONPATH=/home/junsung/cross_modal_tactile/tactile_benchmark CUDA_VISIBLE_DEVICES=0 python tactile_policy/main.py --batch_size 8 --num_epochs 50000 --lr 1e-5 --exptid hammer_eflesh_latent_static12 --dataset_json hammer_eflesh_latent_static12 --model_cfg act_cross_modal_pnp_z16 --chunk_size 64 2>&1 | tee train_logs/hammer_eflesh_latent_static12.log; exec bash"

tmux kill-session -t hammer_flexi_latent_static12 2>/dev/null; tmux new-session -d -s hammer_flexi_latent_static12 "source /home/junsung/anaconda3/etc/profile.d/conda.sh && conda activate tactile_bench && cd ~/cross_modal_tactile/tactile_benchmark && PYTHONPATH=/home/junsung/cross_modal_tactile/tactile_benchmark CUDA_VISIBLE_DEVICES=0 python tactile_policy/main.py --batch_size 8 --num_epochs 50000 --lr 1e-5 --exptid hammer_flexi_latent_static12 --dataset_json hammer_flexi_latent_static12 --model_cfg act_cross_modal_pnp_z16 --chunk_size 64 2>&1 | tee train_logs/hammer_flexi_latent_static12.log; exec bash"
Note: 30 train episodes is small — if you see overfitting/mode collapse, drop --chunk_size to 16 (per the README tuning note for small sets).