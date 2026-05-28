# Training & Evaluation — `dataset_new` (10 objects)

Dataset entry: `dataset_new` in `config/datasets_config_sachin.yaml`.
Data root: `tactile_benchmark/data/latent_data_new_processed`.
Objects (10): `circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle`.
Split: episode-level, per-object val ids (~20% val, evenly sampled across the
front-half / back-half halves of the recording order so contact distribution
matches train).

`OBJECTS` is reused below for brevity.

```bash
OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"
```

---

## Model configs & which command uses which

All configs live in `config/models_config.yaml`. Every command except `clr_only`
uses **`cross_modal_vae3`** (3 encoders + 3 decoders + KL). The losses differ, not
the architecture.

| Command | Script | `model_name` | Loss class | Alignment term |
|---|---|---|---|---|
| Our method (infonce_recon) | `train_static_sachin.py` | `cross_modal_vae3` | `CrossModalVAE3Loss` | InfoNCE(μ) |
| Recon only | `train_static_sachin.py` | `cross_modal_vae3` | `CrossModalVAE3Loss` | InfoNCE off (`--infonce_warmup_start 9999`) |
| Contrastive only (clr_only) | `train_static_sachin.py` | `cross_modal_clr3` | `CrossModalVAE3CLRLoss` | NT-Xent only (no recon/KL) |
| **Uniforce** | `train_uniforce_sachin.py` | **`cross_modal_vae3`** | `CrossModalVAE3UniforceLoss` | **L2 pull** (`loss: 'uniforce'` injected) |
| Anchor / slow | `train_slow_anchor_sachin.py` | `cross_modal_vae3` | `CrossModalVAE3Loss` + L_slow/L_anchor | InfoNCE(μ) + temporal terms |

### Key hyperparameters — `cross_modal_vae3` (the default model config)

| Param | Value | Meaning |
|---|---|---|
| `projection_dim` | 16 | Shared latent dim (override `--projection_dim`, adds `_pd<N>` to run name) |
| `batch_size` | 64 | |
| `learning_rate` | 1e-4 | Override `--lr` |
| `epochs` | 100 | Override `--epochs` |
| `optimizer` / `weight_decay` | Adam / 1e-4 | |
| `seed` | 42 | |
| `resnet` | resnet18 | Daimon image encoder |
| `magnetic_input_dim` / `magnetic_hidden_dims` | 15 / [64,128,256] | eFlesh MLP encoder |
| `flexitac_arch` | residual | FlexiTac encoder/decoder (new runs; not checkpoint-compatible with `small_cnn`) |
| `lambda_recon` | 1.0 | Weight on 12-branch L1 recon |
| `beta_max` / `kl_warmup_epochs` | 0.1 / 30 | KL weight β ramps 0→0.1 over 30 epochs |
| `temperature` | 0.01 | InfoNCE temperature |
| `infonce_warmup_start` / `infonce_warmup_epochs` | 0 / 0 | Alignment weight α stays 0 until `start`, then ramps 0→1 over `epochs`. Override `--infonce_warmup_start` / `--infonce_warmup_epochs` |

Notes:
- **Uniforce** reuses every value above; `loss: 'uniforce'` only changes the alignment
  term to an L2 pull between matched paired means. α still follows the same
  `infonce_warmup_*` curriculum. A pure L2 pull has no negatives, so keep
  `lambda_recon` healthy or delay the pull with `--infonce_warmup_start` to avoid collapse.
- **Recon only** = same config, alignment never turns on (`--infonce_warmup_start 9999`).

### Key hyperparameters — `cross_modal_clr3` (contrastive-only baseline)

Same three encoders, but **no decoders / no KL**. `projection_dim` 16, `batch_size`
64, `learning_rate` 1e-4, `optimizer` Adam, `weight_decay` 1e-4, `seed` 42.
`temperature` is **0.03** (NT-Xent), applied at full weight from epoch 0 — so
`lambda_recon` / `beta` / `infonce_warmup_*` do not apply.

---

## Our method — InfoNCE + Recon + KL

Model config: `cross_modal_vae3` (defaults above). Loss = InfoNCE(μ) + recon + KL.

```bash
OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"
CUDA_VISIBLE_DEVICES=2 python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name infonce_recon \
    --objects $OBJECTS \
    --epochs 300 \
    --num_workers 32

```

```bash
OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"
CUDA_VISIBLE_DEVICES=0 python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name infonce_recon \
    --objects $OBJECTS \
    --epochs 100 \
    --debiased \
    --contact_label_root contact_labels/dataset_new
```

## Baseline 1 — Recon Only (no alignment)

Model config: `cross_modal_vae3`. `--infonce_warmup_start 9999` holds α=0 the whole
run, so only recon + KL train.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name recon_only \
    --objects $OBJECTS \
    --epochs 300 \
    --infonce_warmup_start 9999
```

## Baseline 2 — Contrastive Only (no recon, no KL)

Model config: `cross_modal_clr3` (no decoders, no KL). Loss = NT-Xent only,
`temperature` 0.03, full weight from epoch 0.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --model_name cross_modal_clr3 \
    --device cuda:0 \
    --run_name clr_only \
    --objects $OBJECTS \
    --epochs 100
```

## Baseline 3 — Uniforce (L2 pull + Recon + KL)

Model config: `cross_modal_vae3` (same as Our method); the script injects
`loss: 'uniforce'`, swapping InfoNCE for an L2 pull. No negatives → collapse risk, so
optionally delay the pull with `--infonce_warmup_start <N>`.

```bash
OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"
CUDA_VISIBLE_DEVICES=0 python scripts/train_uniforce_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name uniforce \
    --objects $OBJECTS \
    --epochs 300
```

## Anchor / Slow (InfoNCE + Recon + KL + temporal terms)

Model config: `cross_modal_vae3` (same as Our method) plus two temporal losses on the
dynamic data. `--lambda_slow` weights L_slow (‖μ_t−μ_{t+1}‖², makes an episode's frames
similar); `--lambda_anchor` weights L_anchor (‖μ_nocontact‖², pulls no-contact frames to
0). Both need `--dynamic_dataset_name dataset_cross_modal_tri_dynamic_sachin_new`
(its default points at the *padded* dataset, which is wrong for `dataset_new`).

```bash
OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"

CUDA_VISIBLE_DEVICES=0 python scripts/train_slow_anchor_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name anchor \
    --objects $OBJECTS \
    --epochs 100 \
    --lambda_slow 4 \
    --lambda_anchor
```


---

## Evaluation

```bash
OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cross_modal_vae3_new.py \
    --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_new_run_infonce_recon_db_20260524_172250 \
    --dataset_name dataset_new \
    --objects $OBJECTS \
            --dynamic_data_root latent_data_processed_dynamic/pnp latent_data_processed_dynamic/wipe latent_data_processed_dynamic/plug\
        --dynamic_objects pnp wipe plug\ 
    --contact_label_root contact_labels/dataset_new \
    --device cuda:0
```

## Contact label build (one-time, before any eval that uses the contact probe)

```bash
python scripts/build_contact_labels.py \
    --data_root tactile_benchmark/data/latent_data_new_processed \
    --out_root contact_labels/dataset_new \
    --objects $OBJECTS \
    --visualize
```