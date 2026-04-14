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

```bash
python tactile_policy/main.py \
  --batch_size 8 \
  --num_epochs 50000 \
  --lr 1e-4 \
  --chunk_size 64 \
  --exptid pnp_flexi_raw_mlp \
  --dataset_json pnp_flexi \
  --model_cfg act_flexi \
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
