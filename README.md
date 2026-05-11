# Cross-Modal Tactile Representation Learning

Cross-modal variational self-supervised learning for tactile sensing across heterogeneous sensor types.  
Supports two- and three-sensor configurations sharing a common variational latent space trained with InfoNCE + reconstruction + KL divergence.

## Architecture

### CrossModalVAE (2-sensor: Daimon + eFlesh)

```
Daimon (3, 224, 224) → ResNet-18 → h_i (512) → VAE head → (μ_i, logσ²_i) → z_i (16)
eFlesh (15,)         → MLP       → h_j (512) → VAE head → (μ_j, logσ²_j) → z_j (16)
```

**Loss:**

```
L = L_InfoNCE(μ_i, μ_j)  +  λ_recon · L_recon(z_i, z_j)  +  β · L_KL(μ, σ²)
```

### CrossModalVAE3 (3-sensor: Daimon + eFlesh + FlexiTac)

```
Daimon   (3, 224, 224) → ResNet-18        → VAE head → (μ_D, logσ²_D) → z_D (16)
eFlesh   (15,)         → MLP              → VAE head → (μ_E, logσ²_E) → z_E (16)
FlexiTac (1, 12, 16)   → small CNN        → VAE head → (μ_F, logσ²_F) → z_F (16)
```

Three pairwise losses (DE, EF, FD) are optimised jointly. The unobserved DF alignment emerges transitively through the shared N(0, I) prior.

**Loss (per step):**

```
L = α · Σ_{p∈{DE,EF,FD}} L_InfoNCE(μ_a^p, μ_b^p)
  + λ_recon · Σ_p L_recon^p
  + β · Σ_p L_KL^p
```

**Reconstruction-first curriculum (InfoNCE delayed warm-up):**  
α stays 0 until `infonce_warmup_start`, then linearly ramps 0 → 1 over `infonce_warmup_epochs`. This lets each encoder first learn to pack contact information into the latent space via reconstruction before cross-modal alignment is introduced, preventing early contrastive collapse.

| Epoch range | β (KL) | α (InfoNCE) | Dominant signal |
| --- | --- | --- | --- |
| 0 → `kl_warmup_epochs` | 0 → β_max | 0 | Reconstruction only |
| `kl_warmup_epochs` → `infonce_warmup_start` | β_max | 0 | Recon + KL |
| `infonce_warmup_start` → `+infonce_warmup_epochs` | β_max | 0 → 1 | Recon + KL + InfoNCE |
| thereafter | β_max | 1 | All losses |


| Term                         | Applied to         | Role                                          |
| ---------------------------- | ------------------ | --------------------------------------------- |
| InfoNCE (NT-Xent)            | μ (mean)           | Stable cross-modal alignment                  |
| L1 reconstruction (4-branch) | z (sampled latent) | Forces z to be a shared generative bottleneck |
| KL divergence                | μ, logσ²           | Regularizes latent space toward N(0, I)       |


**4-branch reconstruction (per pair):**

- `D_a(z_a) → x_a` (self-recon)
- `D_a(z_b) → x_a` (cross-recon: partner z decodes into modality A)
- `D_b(z_b) → x_b` (self-recon)
- `D_b(z_a) → x_b` (cross-recon: modality A z decodes into modality B)

**KL warm-up:** β linearly increases from 0 → `beta_max` over `kl_warmup_epochs` to prevent posterior collapse during early training.

## Sensors & Data


| Sensor   | Type                          | Raw output                  | Processed                           |
| -------- | ----------------------------- | --------------------------- | ----------------------------------- |
| Daimon   | Vision-based tactile          | Image (240×320), B-channel  | Resized 224×224, z-score normalized |
| eFlesh   | Magnetic (5 magnets × 3 axes) | Vector (15,) ~[-2000, 2000] | Per-channel z-score normalized      |
| FlexiTac | Resistive array               | Matrix (12×16)              | Per-channel z-score normalized      |


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

### CrossModalVAE3 (3-sensor: Daimon + eFlesh + FlexiTac, recommended)

```bash
# All objects (flat layout)
python scripts/train_model.py \
    --model_name cross_modal_vae3 \
    --dataset_name dataset_cross_modal_tri \
    --device cuda:0 \
    --run_name tri_minimal

# Select specific objects (per-object layout)
python scripts/train_model.py \
    --model_name cross_modal_vae3 \
    --dataset_name dataset_cross_modal_tri_object \
    --device cuda:0 \
    --run_name tri_cyl_plug \
    --objects cylinder plug
```

Checkpoints saved to:

```
checkpoints/cross_modal/cross_modal_vae3/dataset_cross_modal_tri{_object}_run_{run_name}_{timestamp}/
  model.pt
  model_details.yaml
```

#### 2-Stage training with dynamic OOD pretrain

For OOD (e.g. dynamic motion) inference, the encoder can be aligned to the
dynamic distribution first via self-reconstruction on **unpaired** dynamic data
(`latent_data_processed_object/dynamic_data/{daimon,eflesh,flexi}/`), then
fine-tuned on the paired data with the full contrastive + reconstruction loss.

```bash
# Stage 1: dynamic self-recon pretrain (no InfoNCE, no cross-recon)
python scripts/train_model.py \
    --model_name cross_modal_vae3 \
    --dataset_name dataset_cross_modal_tri_dynamic \
    --phase dynamic_pretrain \
    --device cuda:0 \
    --run_name dyn_pretrain

# Stage 2: paired full training, initialised from Stage 1 weights
python scripts/train_model.py \
    --model_name cross_modal_vae3 \
    --dataset_name dataset_cross_modal_tri_object \
    --phase paired \
    --resume_from checkpoints/cross_modal_tri_dynamic/cross_modal_vae3/dataset_cross_modal_tri_dynamic_run_dyn_pretrain_<ts>/model.pt \
    --objects cylinder plug \
    --device cuda:0 \
    --run_name tri_cyl_plug_dynpre
```

Stage-1 hyperparameters (`config/models_config.yaml`):

| Parameter                  | Default | Description                                                       |
| -------------------------- | ------- | ----------------------------------------------------------------- |
| `dynamic_epochs`           | 50      | Number of Stage 1 epochs                                          |
| `dynamic_beta_max`         | 0.1     | KL weight after warm-up (set 0 to disable KL during Stage 1)      |
| `dynamic_kl_warmup_epochs` | 10      | Linear KL warm-up window for Stage 1                              |

Stage 1 trains only `L = λ_recon · (L_D→D + L_E→E + L_F→F) + β · (KL_D + KL_E + KL_F)`. No InfoNCE, no cross-recon. Stage 2 resumes the full curriculum (`α` InfoNCE warm-up, `β` KL warm-up, 4-branch per-pair recon) using the Stage 1 weights as initialisation.

### CrossModalVAE (2-sensor: Daimon + eFlesh)

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

#### CrossModalVAE3

| Parameter               | Default | Description                                                        |
| ----------------------- | ------- | ------------------------------------------------------------------ |
| `lambda_recon`          | 1.0     | Weight for 4-branch L1 reconstruction loss (×3 pairs)             |
| `beta_max`              | 0.1     | Maximum KL weight after warm-up                                    |
| `kl_warmup_epochs`      | 30      | Epochs to linearly ramp β from 0 → `beta_max`                     |
| `infonce_warmup_start`  | 50      | Epoch at which InfoNCE weight α begins to ramp up (0 before this) |
| `infonce_warmup_epochs` | 50      | Epochs to linearly ramp α from 0 → 1                              |
| `temperature`           | 0.01    | NT-Xent temperature applied to μ vectors                          |
| `projection_dim`        | 16      | Shared latent space dimension                                      |
| `batch_size`            | 64      | Training batch size                                                |
| `learning_rate`         | 1e-4    | Adam optimizer learning rate                                       |

#### CrossModalVAE

| Parameter          | Default | Description                                                     |
| ------------------ | ------- | --------------------------------------------------------------- |
| `lambda_recon`     | 1.0     | Weight for 4-branch L1 reconstruction loss                      |
| `beta_max`         | 0.01    | Maximum KL weight after warm-up (too high → posterior collapse) |
| `kl_warmup_epochs` | 30      | Epochs to linearly ramp β from 0 → `beta_max`                   |
| `temperature`      | 0.03    | NT-Xent temperature applied to μ vectors                        |
| `projection_dim`   | 16      | Latent space dimension (μ, z)                                   |
| `batch_size`       | 64      | Training batch size                                             |
| `learning_rate`    | 1e-4    | Adam optimizer learning rate                                    |


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


| Parameter      | Grid                  |
| -------------- | --------------------- |
| `lambda_recon` | 0.1, 0.5, 1.0, 5.0    |
| `beta_max`     | 0.001, 0.01, 0.1      |
| `temperature`  | 0.01, 0.03, 0.07, 0.1 |


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

