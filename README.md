# TactX

TactX learns a shared tactile latent space across heterogeneous tactile sensors. The code in this repository trains and evaluates the cross-sensor representation model described in [TACTX.pdf](TACTX.pdf), including the reconstruction-only, contrastive-only, and L2-alignment baselines used for comparison.

The main release commands use the 10-object paired dataset entry `dataset_new` in `config/datasets_config_sachin.yaml`.

## Installation

Run commands from the repository root.

```bash
conda create -n tactx python=3.10 -y
conda activate tactx
pip install -r requirements.txt
```

The training scripts log to Weights & Biases. Use `WANDB_MODE=offline` in the commands below for local/offline runs.

## Dataset / Checkpoints

The source data and trained checkpoints are intentionally not tracked by Git. Place them in the following repo-relative locations:

```text
tactile_benchmark/data/latent_data_new_processed/
checkpoints/cross_modal_tri/cross_modal_vae3/
checkpoints/cross_modal_tri/cross_modal_clr3/
```

The `dataset_new` config expects paired data under:

```text
tactile_benchmark/data/latent_data_new_processed/{de_pair,ef_pair,fd_pair}/{sensor}/{object}/ep*_fr*.pt
```

Checkpoint directories used for evaluation should contain `model_details.yaml` and either `model_ep100.pt` or `model.pt`.

Define the object list once per shell:

```bash
OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"
```

## Training

Train the TactX objective: InfoNCE alignment on posterior means, self/cross reconstruction, and KL regularization.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name infonce_recon \
    --objects $OBJECTS \
    --epochs 300 \
    --num_workers 32
```

Optional contact-label-aware variant:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name infonce_recon \
    --objects $OBJECTS \
    --epochs 100 \
    --debiased \
    --contact_label_root contact_labels/dataset_new
```

## Evaluation

First build contact labels if you want the contact probe:

```bash
python scripts/build_contact_labels.py \
    --data_root tactile_benchmark/data/latent_data_new_processed \
    --out_root contact_labels/dataset_new \
    --objects $OBJECTS \
    --visualize
```

Evaluate a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cross_modal_vae3_new.py \
    --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_new_run_infonce_recon_db_20260524_172250 \
    --dataset_name dataset_new \
    --objects $OBJECTS \
    --contact_label_root contact_labels/dataset_new \
    --out_dir eval_results/dataset_new_infonce_recon \
    --device cuda:0
```

Add dynamic evaluation data when available:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cross_modal_vae3_new.py \
    --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_new_run_infonce_recon_db_20260524_172250 \
    --dataset_name dataset_new \
    --objects $OBJECTS \
    --dynamic_data_root latent_data/latent_data_processed_dynamic/pnp latent_data/latent_data_processed_dynamic/wipe latent_data/latent_data_processed_dynamic/plug \
    --dynamic_objects pnp wipe plug \
    --contact_label_root contact_labels/dataset_new \
    --out_dir eval_results/dataset_new_infonce_recon_dynamic \
    --device cuda:0
```

For a contrastive-only checkpoint, add `--contrastive_only` and point `--checkpoint_dir` to the `cross_modal_clr3` run.

## Baselines

Reconstruction only:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name recon_only \
    --objects $OBJECTS \
    --epochs 300 \
    --infonce_warmup_start 9999
```

Contrastive only:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --model_name cross_modal_clr3 \
    --device cuda:0 \
    --run_name clr_only \
    --objects $OBJECTS \
    --epochs 100
```

L2 alignment / Uniforce-style baseline:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python scripts/train_uniforce_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name uniforce \
    --objects $OBJECTS \
    --epochs 300
```

Optional temporal slow/anchor baseline:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python scripts/train_slow_anchor_sachin.py \
    --dataset_name dataset_new \
    --dynamic_dataset_name dataset_cross_modal_tri_dynamic_sachin_new \
    --device cuda:0 \
    --run_name anchor \
    --objects $OBJECTS \
    --epochs 100 \
    --lambda_slow 4.0 \
    --lambda_anchor 1.0
```

## Visualization

Evaluation writes metrics and figures under `--out_dir`, including cosine distributions, t-SNE plots, reconstruction grids, and optional dynamic overlays. These generated outputs are ignored by Git.

For quick plot generation during development, reduce t-SNE cost:

```bash
python scripts/eval_cross_modal_vae3_new.py \
    --checkpoint_dir <checkpoint_dir> \
    --dataset_name dataset_new \
    --objects $OBJECTS \
    --out_dir eval_results/smoke_eval \
    --max_tsne_points 200 \
    --device cuda:0
```

## Repository Structure

```text
config/                    Dataset and model configuration files
docs/                      Additional training notes
joint_embedding_learning/  TactX models, losses, datasets, and training loops
scripts/                   Training, evaluation, preprocessing, and visualization scripts
tactile_benchmark/         Downstream tactile-policy utilities and configs
```

Local datasets, checkpoints, W&B logs, contact labels, and generated figures are ignored by `.gitignore`.

## Citation

If you use this code, please cite the paper. Replace the placeholder fields once the final citation is available.

```bibtex
@inproceedings{tactx2026,
  title     = {TactX: Learning Shared Tactile Representations Across Diverse Sensors},
  author    = {Anonymous Authors},
  booktitle = {Conference on Robot Learning},
  year      = {2026}
}
```

