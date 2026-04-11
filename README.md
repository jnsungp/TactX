# Cross-modal Tactile Transfer

Cross-modal contrastive self-supervised learning for tactile sensing across heterogeneous sensor types.  
We pair a **Daimon** (vision-based tactile image sensor) and an **eFlesh** (magnetic tactile sensor) mounted on the same gripper, and learn a shared latent representation using NT-Xent contrastive loss.

## Method Overview

- Positive pairs: same contact frame captured by Daimon and eFlesh simultaneously
- Separate encoders per modality: ResNet-18 (image) + MLP (magnetic vector)
- Shared projection heads map both modalities into a 64-dim latent space
- NT-Xent (SimCLR) loss aligns the two modalities in the latent space

## Data

### Sensors
| Sensor | Type | Output |
|---|---|---|
| Daimon | Vision-based tactile | Image (480×640×3) → B-channel crop (240×320) |
| eFlesh | Magnetic (5 magnets × 3 axes) | Vector (15,) |

### Dataset structure
Raw data: `tactile_data_new/left_daimon_right_eflesh_{object_name}/episode_XXXX/*.pkl`  
Each `.pkl` file contains one timestep: `{ts, s0, s1, img0, img1}`

### Preprocessing
Contact filtering (Daimon B-ch mean > 0.005) and Daimon horizontal mirror flip (gripper symmetry correction).

```bash
python scripts/preprocess_cross_modal.py

# Options
python scripts/preprocess_cross_modal.py --contact_threshold 0.01
python scripts/preprocess_cross_modal.py --no_contact_filter   # keep all frames
python scripts/preprocess_cross_modal.py --no_mirror           # disable flip
```

Processed output: `processed_data/cross_modal/{daimon,eflesh}/{object_name}/*.pt`

## Installation

Install SimCLR package:
```bash
pip install simclr
```

Install T3 package (for T3 baseline):
```bash
git clone https://github.com/alanzjl/t3
cd t3 && pip install -e .
```

## Training

```bash
python scripts/train_model.py --model_name cross_modal --device cuda:0 --dataset_name dataset_cross_modal

# With a custom run name (timestamp is auto-appended)
python scripts/train_model.py --model_name cross_modal --device cuda:0 --dataset_name dataset_cross_modal --run_name exp1
```

Checkpoints are saved to: `checkpoints/cross_modal/cross_modal/dataset_cross_modal_run_{run_name}_{timestamp}/`

### Baselines (same training script)
```bash
python scripts/train_model.py --model_name simclr        --device cuda:0 --dataset_name dataset_1
python scripts/train_model.py --model_name barlow_twins  --device cuda:0 --dataset_name dataset_1
```

### Model config
All hyperparameters are in `config/models_config.yaml` and dataset paths in `config/datasets_config.yaml`.

## Evaluation

Run `eval_cross_modal.ipynb` for:
- **t-SNE** visualization of the shared latent space (modality alignment, episode clustering)
- **Cosine similarity** distribution: positive vs. negative pairs
- **Cross-modal retrieval accuracy**: Top-1/5/10 Daimon→eFlesh and eFlesh→Daimon
