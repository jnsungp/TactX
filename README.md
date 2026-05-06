# Cross-Modal Tactile Representation Learning

Cross-modal variational self-supervised learning for tactile sensing across heterogeneous sensor types.  
We pair a **Daimon** (vision-based tactile image sensor) and an **eFlesh** (magnetic tactile sensor) mounted on the same gripper, and learn a shared variational latent space using a combined InfoNCE + reconstruction + KL divergence loss.

## Architecture: CrossModalVAE

```
Daimon (3, 224, 224) → ResNet-18 → h_i (512) → VAE head → (μ_i, logσ²_i) → z_i (16)
eFlesh (15,)         → MLP       → h_j (512) → VAE head → (μ_j, logσ²_j) → z_j (16)
```

**Loss:**
```
L = L_InfoNCE(μ_i, μ_j)  +  λ_recon · L_recon(z_i, z_j)  +  β · L_KL(μ, σ²)
```

| Term | Applied to | Role |
|---|---|---|
| InfoNCE (NT-Xent) | μ (mean) | Stable cross-modal alignment |
| L1 reconstruction (4-branch) | z (sampled latent) | Forces z to be a shared generative bottleneck |
| KL divergence | μ, logσ² | Regularizes latent space toward N(0, I) |

**4-branch reconstruction:**
- `D_img(z_i) → x_i` (image self-recon)
- `D_img(z_j) → x_i` (image cross-recon, magnetic z decodes to image)
- `D_mag(z_j) → x_j` (magnetic self-recon)
- `D_mag(z_i) → x_j` (magnetic cross-recon, image z decodes to magnetic)

**KL warm-up:** β linearly increases from 0 → `beta_max` over `kl_warmup_epochs` to prevent posterior collapse during early training.

## Sensors & Data

| Sensor | Type | Raw output | Processed |
|---|---|---|---|
| Daimon | Vision-based tactile | Image (240×320), B-channel | Resized 224×224, z-score normalized |
| eFlesh | Magnetic (5 magnets × 3 axes) | Vector (15,) ~[-2000, 2000] | Per-channel z-score normalized |

### Dataset structure
```
processed_data/cross_modal/
  daimon/{object_name}/ep{NNNN}_fr{FFFF}.pt   # {"image": tensor(1,240,320)}
  eflesh/{object_name}/ep{NNNN}_fr{FFFF}.pt   # {"magnetic": tensor(15,)}
```

**Objects (training):** can, cylinder, large_ball, pen, small_ball (total ~30k frames)  
**Validation:** `{object}_val` splits

### Preprocessing
```bash
python scripts/preprocess_cross_modal.py

# Options
python scripts/preprocess_cross_modal.py --contact_threshold 0.01
python scripts/preprocess_cross_modal.py --no_contact_filter   # keep all frames
python scripts/preprocess_cross_modal.py --no_mirror           # disable horizontal flip
```

## Installation

```bash
pip install simclr
```

## Training

### CrossModalVAE (recommended)
```bash
python scripts/train_model.py \
    --model_name cross_modal_vae \
    --dataset_name dataset_cross_modal \
    --device cuda:0

# With a custom run name (timestamp is auto-appended)
python scripts/train_model.py \
    --model_name cross_modal_vae \
    --dataset_name dataset_cross_modal \
    --device cuda:0 \
    --run_name exp1
```

Checkpoints saved to:
```
checkpoints/cross_modal/cross_modal_vae/dataset_cross_modal_run_{run_name}_{timestamp}/
  model.pt
  model_details.yaml
```

### Key hyperparameters (`config/models_config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `lambda_recon` | 1.0 | Weight for 4-branch L1 reconstruction loss |
| `beta_max` | 0.01 | Maximum KL weight after warm-up (too high → posterior collapse) |
| `kl_warmup_epochs` | 30 | Epochs to linearly ramp β from 0 → `beta_max` |
| `temperature` | 0.03 | NT-Xent temperature applied to μ vectors |
| `projection_dim` | 16 | Latent space dimension (μ, z) |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 1e-4 | Adam optimizer learning rate |

### Baselines
```bash
# InfoNCE only (no VAE)
python scripts/train_model.py --model_name cross_modal --device cuda:0 --dataset_name dataset_cross_modal

# SimCLR / Barlow Twins
python scripts/train_model.py --model_name simclr       --device cuda:0 --dataset_name dataset_1
python scripts/train_model.py --model_name barlow_twins --device cuda:0 --dataset_name dataset_1
```

## Hyperparameter Sweep

Sweeps the 3 most critical VAE parameters across 4×3×4 = **48 combinations**:

| Parameter | Grid |
|---|---|
| `lambda_recon` | 0.1, 0.5, 1.0, 5.0 |
| `beta_max` | 0.001, 0.01, 0.1 |
| `temperature` | 0.01, 0.03, 0.07, 0.1 |

```bash
# Dry run — print all 48 combinations without training
python scripts/sweep_cross_modal.py \
    --model_name cross_modal_vae \
    --dry_run

# Full sweep
python scripts/sweep_cross_modal.py \
    --model_name cross_modal_vae \
    --dataset_name dataset_cross_modal \
    --device cuda:0 \
    --project CMTEJ_sweep

# Resume (skip already-checkpointed runs automatically)
python scripts/sweep_cross_modal.py \
    --model_name cross_modal_vae \
    --dataset_name dataset_cross_modal \
    --device cuda:0 \
    --sweep_id sweep_20260506_012600   # reuse existing group tag

# Force re-run everything (ignore existing checkpoints)
python scripts/sweep_cross_modal.py \
    --model_name cross_modal_vae \
    --dataset_name dataset_cross_modal \
    --device cuda:0 \
    --no_skip_existing
```

Each run is logged to wandb under the same `sweep_id` group.  
Use the **parallel coordinates plot** in wandb to identify which `(lambda_recon, beta_max, temperature)` combination gives the best contrastive alignment.

## Evaluation

### Embedding quality (t-SNE, cosine similarity, retrieval)
Run `eval_cross_modal_vae.ipynb`:
- **t-SNE** of μ space and h space
- **Cosine similarity** distribution: positive vs. negative pairs
- **Cross-modal retrieval accuracy**: Top-1/5/10 Daimon→eFlesh and eFlesh→Daimon

### Reconstruction visualization (MP4)
```bash
python scripts/evaluate_cross_modal_vae.py \
    --checkpoint_dir checkpoints/cross_modal/cross_modal_vae/<run_name> \
    --device cuda:0
```

Outputs to `eval_results/cross_modal_vae/`:
- Static PNG grids (ground truth / self-recon / cross-recon)
- Per-episode MP4 videos

### Fair reconstruction loss comparison (VAE vs UniForce)
```bash
# VAE only (canonical [0,1] L1)
python scripts/eval_recon_fair.py \
    --vae_checkpoint_dir checkpoints/cross_modal/cross_modal_vae/<run_name> \
    --device cuda:0

# Side-by-side comparison with UniForce
python scripts/eval_recon_fair.py \
    --vae_checkpoint_dir checkpoints/cross_modal/cross_modal_vae/<run_name> \
    --uniforce_checkpoint /path/to/uniforce_final.pt \
    --device cuda:0
```

Both models' L1 losses are converted to a canonical **[0, 1] min-max space** before comparison (raw z-score and min-max normalizations differ between models).

### Latent conversion (for downstream policy)
```bash
python scripts/convert_tac_to_latent.py \
    --model_name cross_modal_vae \
    --latent_type mu \   # mu | z | h
    --device cuda:0
```
