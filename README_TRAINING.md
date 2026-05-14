# Joint cyl+plug dynamic-align training

Latest workflow. One tri-modal encoder (Daimon + eFlesh + FlexiTac) trained on
paired static **and** unpaired dynamic data, pooled across cylinder + plug.

## Data layout (must look like this)

```
latent_data_processed_object/
├── de_pair/{daimon,eflesh}/{cylinder,plug}/ep****_fr****.pt
├── ef_pair/{eflesh,flexi}/{cylinder,plug}/ep****_fr****.pt
├── fd_pair/{flexi,daimon}/{cylinder,plug}/ep****_fr****.pt
└── dynamic_data/
    ├── cylinder/{daimon,eflesh,flexi}/ep****_fr****.pt
    └── plug/{daimon,eflesh,flexi}/ep****_fr****.pt
```

## Normalization stats (single pooled block per regime)

| Regime  | Config file                          | Dataset entry                            |
|---------|--------------------------------------|------------------------------------------|
| Static  | `config/datasets_config_sachin.yaml` | `dataset_sachin`                         |
| Dynamic | `config/datasets_config.yaml`        | `dataset_cross_modal_tri_dynamic_all`    |

Both use cyl+plug pooled mean/std (computed by
`scripts/compute_per_object_stats.py` → `per_object_stats.yaml` keys
`combined_static`, `combined_dynamic`).

The dynamic entry's `data_root` is a list — the loader concatenates the two
object subdirs (`get_dynamic_self_datasets` accepts `list` data_root and uses
`ConcatDataset`).

## Recompute stats (rarely needed — only if data changes)

```bash
cd ~/cross_modal_tactile   # conda activate cmt
python scripts/compute_per_object_stats.py
# writes per_object_stats.yaml; paste combined_static / combined_dynamic
# back into the two config files above.
```

## Training commands (plain + mmd-only, sequential in tmux)

```bash
cd ~/cross_modal_tactile

tmux new-session -d -s dynalign_joint -c ~/cross_modal_tactile
tmux send-keys -t dynalign_joint "source ~/.bashrc && conda activate cmt && \
CUDA_VISIBLE_DEVICES=0 python scripts/train_dynamic_align_sachin.py \
    --device cuda:0 --run_name dynalign_plain_cyl_plug \
    --objects cylinder plug \
    --dynamic_dataset_name dataset_cross_modal_tri_dynamic_all \
    --gamma_mmd_max 0 --delta_cycle_max 0 ; \
CUDA_VISIBLE_DEVICES=0 python scripts/train_dynamic_align_sachin.py \
    --device cuda:0 --run_name dynalign_mmd_only_cyl_plug \
    --objects cylinder plug \
    --dynamic_dataset_name dataset_cross_modal_tri_dynamic_all \
    --gamma_mmd_max 0.5 --delta_cycle_max 0" Enter
```

- Attach: `tmux attach -t dynalign_joint`
- Detach: `Ctrl-b d`
- Loss-weight knobs: `--gamma_mmd_max` (MMD bridge), `--delta_cycle_max` (cycle).
  Both default to 0.5 in `training_dynamic_align.py`; set to 0 to disable.

Checkpoints land in:
```
checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_<run_name>_<TS>/
```

## Tac→latent conversion (teleop HDF5 → 16-d µ)

The converter loads stats from the checkpoint's saved `dataset_details.yaml`
(**static** stats) by default. For teleop data — which lives in the dynamic
regime — pass `--stats_dataset_name dataset_cross_modal_tri_dynamic_all` to
override with dynamic stats.

```bash
CKPT=checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_run_dynalign_mmd_only_cyl_plug_<TS>

# eFlesh teleop (plug pnp)
python scripts/convert_tac_to_latent_sachin.py \
    --checkpoint_dir "$CKPT" --sensor E \
    --stats_dataset_name dataset_cross_modal_tri_dynamic_all \
    --input_dir  tactile_benchmark/data/recordings/processed/eflesh_eflesh_object_pnp \
    --output_dir tactile_benchmark/data/recordings/processed/eflesh_eflesh_object_pnp_latent

# eFlesh teleop val
python scripts/convert_tac_to_latent_sachin.py \
    --checkpoint_dir "$CKPT" --sensor E \
    --stats_dataset_name dataset_cross_modal_tri_dynamic_all \
    --input_dir  tactile_benchmark/data/recordings/processed/eflesh_eflesh_object_pnp_val \
    --output_dir tactile_benchmark/data/recordings/processed/eflesh_eflesh_object_pnp_val_latent

# FlexiTac teleop (board wipe)
python scripts/convert_tac_to_latent_sachin.py \
    --checkpoint_dir "$CKPT" --sensor F \
    --stats_dataset_name dataset_cross_modal_tri_dynamic_all \
    --input_dir  tactile_benchmark/data/recordings/processed/flexi_flexi_board_wipe \
    --output_dir tactile_benchmark/data/recordings/processed/flexi_flexi_board_wipe_latent

# Daimon teleop
python scripts/convert_tac_to_latent_sachin.py \
    --checkpoint_dir "$CKPT" --sensor D \
    --stats_dataset_name dataset_cross_modal_tri_dynamic_all \
    --input_dir  tactile_benchmark/data/recordings/processed/daimon_daimon_object_pnp \
    --output_dir tactile_benchmark/data/recordings/processed/daimon_daimon_object_pnp_latent
```

`--inference_transform {none, fliplr, flipud, rot180}` is available if teleop
orientation differs from the canonical (sachin per-pair rot180) — start with
`none` and only set it if the latents look off.

The converter prints `[stats override] using dataset_cross_modal_tri_dynamic_all for *_stats`
when the flag takes effect — confirm this in the output.

## Policy training on the new latents

Drop in your existing `pnp_*_latent.json` dataset_json (sensor config
`cross_modal_latent_z16` already has `tactile_dim: 16`, `modality: array`):

```bash
cd ~/cross_modal_tactile/tactile_benchmark   # conda activate tactile_bench

CUDA_VISIBLE_DEVICES=0 python tactile_policy/main.py \
    --batch_size 8 --num_epochs 50000 --lr 1e-5 \
    --exptid pnp_eflesh_cmlatent_cylplug \
    --dataset_json pnp_eflesh_latent \
    --model_cfg cross_modal_latent_z16 \
    --chunk_size 64
```
