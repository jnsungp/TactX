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

## Our method — InfoNCE + Recon + KL

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name infonce_recon \
    --objects $OBJECTS \
    --epochs 300
```

## Baseline 1 — Recon Only (no alignment)

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

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_static_sachin.py \
    --dataset_name dataset_new \
    --model_name cross_modal_clr3 \
    --device cuda:0 \
    --run_name clr_only \
    --objects $OBJECTS \
    --epochs 300
```

## Baseline 3 — Uniforce (L2 pull + Recon + KL)

```bash
OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"
CUDA_VISIBLE_DEVICES=0 python scripts/train_uniforce_sachin.py \
    --dataset_name dataset_new \
    --device cuda:0 \
    --run_name uniforce \
    --objects $OBJECTS \
    --epochs 300
```

---

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cross_modal_vae3_new.py \
    --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/<RUN_DIR> \
    --dataset_name dataset_new \
    --objects $OBJECTS \
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

