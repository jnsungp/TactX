# PCA Encoding for Vision-Based Tactile Sensors

This implementation adds PCA-based encoding as an alternative to ResNet for vision-based tactile sensors (Daimon, Flexitac).

## Motivation

The reviewer asked about representation choices affecting benchmark results. PCA provides a simpler, non-learned baseline for comparison with ResNet encoders.

## Encoding Options

### 1. ResNet (baseline)
- Uses ResNet18 CNN backbone
- Learnable parameters
- Preserves spatial structure
- Config: `sensor_config: daimon`

### 2. PCA
- Linear dimensionality reduction
- No learnable parameters
- Compresses to 256 components
- Config: `sensor_config: daimon_pca`

### 3. PCA + MLP
- PCA compression + learnable MLP
- Middle ground between PCA and ResNet
- Config: `sensor_config: daimon_pca_mlp`

## Usage

### Step 1: Precompute PCA Transformers (Recommended)

Precompute PCA on all training data for consistent representation across tasks:

```bash
# For Daimon sensor
bash bash_scripts/precompute_pca_daimon.sh

# For Flexitac sensor
bash bash_scripts/precompute_pca_flexitac.sh
```

This creates: `/data/pca_transformers/{sensor}_pca_256.pkl`

### Step 2: Train with PCA Encoding

```bash
# Using precomputed PCA
python tactile_policy/main.py \
  --batch_size 16 \
  --num_epochs 50000 \
  --lr 1e-5 \
  --exptid pnp_daimon_pca \
  --dataset_json pnp_daimon \
  --model_cfg act_daimon_pca \
  --chunk_size 64 \
  --pca_path data/pca_transformers/daimon_pca_256.pkl

# Without precomputed PCA (fits on current task data)
python tactile_policy/main.py \
  --batch_size 16 \
  --num_epochs 50000 \
  --lr 1e-5 \
  --exptid pnp_daimon_pca \
  --dataset_json pnp_daimon \
  --model_cfg act_daimon_pca \
  --chunk_size 64
```

### Step 3: Compare Encodings

Run ablation study:

```bash
# Baseline: ResNet
--model_cfg act_daimon

# PCA
--model_cfg act_daimon_pca --pca_path data/pca_transformers/daimon_pca_256.pkl

# PCA + MLP
--model_cfg act_daimon_pca_mlp --pca_path data/pca_transformers/daimon_pca_256.pkl
```

## Architecture Comparison

**ResNet18:**
```
Tactile image (3, 240, 320)
  ↓ ResNet18 backbone (~11M params)
  ↓ Feature maps (512, H, W)
  ↓ Conv2d projection
  → Spatial tokens (many)
```

**PCA:**
```
Tactile image (3, 240, 320)
  ↓ Flatten (230,400)
  ↓ PCA transform (256) [no params]
  ↓ Linear (256 → 512)
  → Single token per sensor
```

**PCA + MLP:**
```
Tactile image (3, 240, 320)
  ↓ Flatten (230,400)
  ↓ PCA transform (256) [no params]
  ↓ MLP [256 → 256 → 512]
  → Single token per sensor
```

## Configuration Files

### Sensor Configs
- `configs/sensors/daimon_pca.yaml` - PCA with 256 components
- `configs/sensors/daimon_pca_mlp.yaml` - PCA + MLP
- `configs/sensors/flexitac_pca.yaml` - PCA for Flexitac

### Model Configs
- `configs/models/act_daimon_pca.yaml`
- `configs/models/act_daimon_pca_mlp.yaml`

## Custom PCA Configuration

Adjust PCA components in sensor config:

```yaml
# configs/sensors/daimon_pca_custom.yaml
sensor_type: "daimon"
modality: "image"
tactile_dim: 0

tactile_image_encoding: "pca_mlp"
pca_n_components: 512  # More components
pca_mlp_hidden_layers: [512, 256]  # Deeper MLP
```

## Notes

- PCA transformers are saved in `dataset_stats.pkl` for inference
- Each sensor (tac0, tac1) has its own PCA transformer
- Precomputed PCA ensures consistent encoding across all experiments
- PCA encoding produces per-sensor tokens (like array modality) not spatial features
