# Update — uniforce loss, static-only trainer, 10-object dataset, sweeps

Changes since the previous commit. Builds on the dynamic-align / pooled-stats
work documented in `README_TRAINING.md` and `README_NORM_STATS_HISTORY.md`.

## 1. Uniforce loss — L2 alignment instead of contrastive

`CrossModalVAE3UniforceLoss` (in `joint_embedding_learning/utils/losses.py`) is
line-for-line identical to `CrossModalVAE3Loss` except the cross-modal
alignment term:

```
CrossModalVAE3Loss          : L_align = InfoNCE(mu_a, mu_b)   (contrastive, has negatives)
CrossModalVAE3UniforceLoss  : L_align = MSE(mu_a, mu_b)       (plain L2, no negatives)
```

Recon (4× L1 per pair) + KL unchanged. Output dict keys identical, so the
existing tri training / validation / logging code consumes it unchanged.

Selected via `model_details['loss']`: `get_loss` (in `utils/utils.py`) now
reads `'infonce'` (default) vs `'uniforce'`/`'l2'` for `cross_modal_vae3`.

**Collapse note:** InfoNCE resists collapse via negatives; pure L2 does not —
its only anti-collapse force is reconstruction. Keep `lambda_recon` healthy
relative to the alignment weight, and prefer a recon-first curriculum.

## 2. Static-only trainer — `scripts/train_static_sachin.py`

The true lean "plain" baseline. The dynamic-align trainer still runs the full
dynamic forward + MMD every step even at `gamma_mmd_max=0` (zero-weighted term
still builds its graph + holds activations). `train_static_sachin.py` calls
`train_tri` instead — paired-static tri-modal only: recon + KL + InfoNCE over
the de/ef/fd pairs, no dynamic data, no MMD, no cycle.

`scripts/train_uniforce_sachin.py` is the same launcher with `loss: 'uniforce'`.

## 3. New 10-object paired-static dataset

`dataset_sachin_static` in `config/datasets_config_sachin.yaml` →
`latent_data_processed_static/`. Same pair layout + per-pair rot180 as
`dataset_sachin`, but 10 objects:

```
circle, cylinder, ellipse, eraser, large_ball, line, plug, small_ball, spam, triangle
```

Stats are pooled `combined_static` over all 10 objects (computed by the script
below, recorded in `per_object_stats.yaml`).

NOTE: `latent_data_processed_static/` is training data — not committed to git.

## 4. `compute_per_object_stats.py` — additions

- `--data_root` flag — point at `latent_data_processed_static` (or any root).
- Dumps pooled `combined_static` / `combined_dynamic` blocks in addition to
  per-object stats.

## 5. New CLI knobs on the train scripts (additive, defaults unchanged)

`train_static_sachin.py` and `train_uniforce_sachin.py` both accept:

| Flag | Effect | run_name suffix |
|---|---|---|
| `--projection_dim N` | shared latent dim (default 16) | `_pd<N>` |
| `--infonce_warmup_start N` | epochs of recon-only before alignment turns on | `_ws<N>` |
| `--infonce_warmup_epochs N` | ramp width: alignment weight 0→1 | `_we<N>` |
| `--epochs N` | total training epochs (default 100) | — |

## 6. `scripts/plot_dynamic_latents_taskwise.py` — new

Task-wise dynamic-latent analysis. Encodes dynamic data per (object × sensor),
runs per-checkpoint t-SNE coloured by task (cylinder=blue, plug=red, sensor by
marker), and — the rigorous part — fits a logistic-regression
cylinder-vs-plug classifier on the **raw latents** (not t-SNE coords) reporting
5-fold CV accuracy (pooled + per-sensor). Misclassified points are ringed.

---

## Training commands

```bash
cd ~/cross_modal_tactile   # conda activate cmt
OBJ="circle cylinder ellipse eraser large_ball line plug small_ball spam triangle"
```

### Plain (InfoNCE) and uniforce (L2), 10-object static

```bash
python scripts/train_static_sachin.py   --device cuda:0 --run_name plain_static10 \
    --dataset_name dataset_sachin_static --objects $OBJ
python scripts/train_uniforce_sachin.py --device cuda:0 --run_name uniforce_static10 \
    --dataset_name dataset_sachin_static --objects $OBJ
```

### Recompute stats for the 10-object dataset

```bash
python scripts/compute_per_object_stats.py \
    --data_root latent_data_processed_static --splits static --objects $OBJ
# paste combined_static into dataset_sachin_static's *_stats blocks
```

### Latent-dim sweep

```bash
for PD in 16 1 2 8 32 64; do
  python scripts/train_static_sachin.py --device cuda:0 --run_name pdsweep \
      --objects cylinder plug --projection_dim $PD
done
```

### Warmup-curriculum sweep — 4 profiles × {InfoNCE, uniforce}, 150 epochs

Profiles (start = recon-only epochs, we = ramp width):

| profile | ws | we | alignment full by |
|---|---|---|---|
| none   | 0  | 0  | epoch 1  |
| short  | 10 | 10 | epoch 20 |
| medium | 25 | 15 | epoch 40 |
| long   | 50 | 25 | epoch 75 |

```bash
STARTS=(0 10 25 50); EPS=(0 10 15 25)
for i in 0 1 2 3; do
  S=${STARTS[$i]}; E=${EPS[$i]}
  python scripts/train_static_sachin.py   --device cuda:0 --run_name swInfonce \
      --dataset_name dataset_sachin_static --objects $OBJ \
      --epochs 150 --infonce_warmup_start $S --infonce_warmup_epochs $E
  python scripts/train_uniforce_sachin.py --device cuda:0 --run_name swUniforce \
      --dataset_name dataset_sachin_static --objects $OBJ \
      --epochs 150 --infonce_warmup_start $S --infonce_warmup_epochs $E
done
```

150 epochs (not 100) so the `long` profile still gets ~75 post-warmup epochs —
otherwise longer-warmup profiles are confounded with undertraining.

## Eval / plotting

```bash
# Full eval (retrieval, recon, cosine, t-SNE, MMD) per checkpoint
python scripts/eval_cross_modal_vae3_sachin.py \
    --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/<run_dir> \
    --device cuda:0 --dataset_name dataset_sachin_static --objects $OBJ \
    --dynamic_dataset_name dataset_cross_modal_tri_dynamic_all

# Task-wise latent separability across checkpoints
python scripts/plot_dynamic_latents_taskwise.py \
    --checkpoint_dir <run_dir_1> <run_dir_2> ... --device cuda:0 \
    --out dynamic_latents_taskwise.png
```

## Experiments run

- **Latent-dim sweep** — `pdsweep_pd{1,2,8,16,32,64}` (cyl+plug static).
- **Warmup-curriculum sweep** — `sw{Infonce,Uniforce}_ws{0,10,25,50}_we{0,10,15,25}`
  on the 10-object static dataset, 150 epochs.
