# Normalization stats — history of approaches

Map old checkpoint dirs to which normalization scheme they used. Each phase
listed below corresponds to a *snapshot* of `datasets_config.yaml` +
`datasets_config_sachin.yaml`. Stats blocks were edited in place — git log
those two files for the exact diffs.

## Quick-find: what scheme did checkpoint X use?

Look at the timestamp suffix on the checkpoint dir name (`..._<YYYYMMDD>_<HHMMSS>`)
and match it against the **Active dates** column below.

| Phase | Run-name pattern                              | Active dates              | Static stats              | Dynamic stats             |
|-------|-----------------------------------------------|---------------------------|---------------------------|---------------------------|
| 1     | `dataset_sachin_run_dynalign_*_cyl_*`         | through 2026-05-12        | cylinder-static (≈ true)  | **cylinder-static** (bug) |
| 2     | `dataset_sachin_run_dynalign_*_plug_*`        | 2026-05-13 (brief)        | cylinder-static (unchanged)| plug-dynamic (true)       |
| 3     | `dataset_sachin_run_dynalign_*_cyl_plug_*`    | 2026-05-13 onward         | pooled cyl+plug static    | pooled cyl+plug dynamic   |

## Phase 1 — original cyl-only runs (had a bug)

**Run names:** `dynalign_plain_cyl`, `dynalign_mmd_only_cyl`,
`dynalign_cycle_only_cyl`, `dynalign_sachin_cylinder` (the "both" variant).

**Training data:** `--objects cylinder`; dynamic from
`dataset_cross_modal_tri_dynamic` (flat-glob, pointed at cylinder data).

**Static stats** (`dataset_sachin`):
```
daimon mean=[0.091920, -0.043407, -0.045044], std=[0.230129, 0.136415, 0.112977]
eflesh mean=[26.78, -514.70, 116.00, -2023.42, 7.84, 1047.58, 223.11, 2531.49, -1088.11, -21.52, -2981.30, -30.56, -1813.46, -38.18, 218.03]
       std=[786.39, 1407.68, 598.35, 2106.47, 532.40, 1565.88, 654.22, 2274.91, 1613.50, 663.63, 2402.27, 744.13, 2340.60, 331.95, 697.78]
flexitac mean=0.073024, std=0.169825
```
These daimon values match the *true* `cylinder_static` stats to 4 decimals.
The eflesh/flexi values were pooled (multi-object) carryovers from an earlier
dataset — not exact cyl-only.

**Dynamic stats** (`dataset_cross_modal_tri_dynamic`):
**Identical to static block above** — copy-pasted. **This was a bug.** The true
cylinder-dynamic distribution is materially different (e.g. daimon ch0
std=0.319 actual vs 0.230 used; eflesh std roughly 1.5× actual). So Phase-1
encoders were trained with dynamic frames z-scored using static-regime
statistics.

**Bug consequence:** dynamic inputs entered the encoder with mean ≈ +0.1, std
≈ 1.4 instead of the intended (0, 1). Encoder adapted, but the MMD bridge had
to fight an artificial input-scale offset that wasn't a real regime
difference. Phase-1 results are still internally consistent (same stats at
train and inference) — just sub-optimal.

## Phase 2 — plug-only runs (transitional)

**Run names:** `dynalign_plain_plug`, `dynalign_mmd_only_plug` — short-lived;
launched after the Phase-1 dynamic-stats bug was discovered but before the
joint-training plan crystallized. May not exist if training was stopped.

**Training data:** `--objects plug`; dynamic from a new
`dataset_cross_modal_tri_dynamic_plug` entry pointing at
`dynamic_data/plug/`. (`dataset_cross_modal_tri_dynamic`'s data_root was
also made explicit: `dynamic_data/cylinder/`.)

**Static stats** (`dataset_sachin`): **unchanged from Phase 1**
(cylinder-static values). We accepted this for the plug-only run because the
encoder is trained from scratch with consistent stats at train + inference
time — the mismatch with the true plug-static distribution gets absorbed by
the encoder weights. This would NOT be acceptable for joint cyl+plug
training.

**Dynamic stats** (`dataset_cross_modal_tri_dynamic_plug`): recomputed from
plug-only dynamic data. True values are very different from Phase 1:
```
daimon mean=[0.181, 0.217, 0.170], std=[0.325, 0.812, 0.702]
       (ch1/ch2 means ≈ 0.2 — plug teleop has real lateral gel push,
        unique to plug; cyl/static daimon ch1/ch2 means hover near 0.)
eflesh stds are 1.5–3× larger than cyl-dynamic — plug presses harder.
```

The matching cylinder-only dyn block was *also* recomputed at this point and
written into `dataset_cross_modal_tri_dynamic` (its stats finally reflect
cylinder-dynamic instead of the Phase-1 copy-paste).

## Phase 3 — joint cyl+plug (current)

**Run names:** `dynalign_plain_cyl_plug`, `dynalign_mmd_only_cyl_plug`
(+ future cycle/both variants).

**Training data:** `--objects cylinder plug`; dynamic from
`dataset_cross_modal_tri_dynamic_all` whose `data_root` is a list of both
object subdirs — the loader does `ConcatDataset` over per-root datasets.
With per-step MMD batches of 64 from each branch and a roughly 2:1 cyl:plug
ratio, both modes are well represented per batch (~42 cyl + ~22 plug).

**Static stats** (`dataset_sachin`): recomputed as **pooled cyl+plug static**
(`per_object_stats.yaml::combined_static`).
```
daimon mean=[0.0754, -0.0485, -0.0463], std=[0.1998, 0.1397, 0.1192]
flexitac mean=0.0929, std=0.1871
```
(eflesh — see config file.)

**Dynamic stats** (`dataset_cross_modal_tri_dynamic_all`): recomputed as
**pooled cyl+plug dynamic** (`per_object_stats.yaml::combined_dynamic`).
```
daimon mean=[0.1393, 0.0498, 0.0485], std=[0.3226, 0.5287, 0.4529]
flexitac mean=0.0683, std=0.1666
```
(eflesh — see config file.)

**Why pooled, not per-object?** A single multi-object encoder should see the
relative differences between objects (plug pushes harder than cyl) at its
input — per-object z-scoring would erase that information and hurt
representation learning. Pooled stats are also what makes the MMD bridge
work: a *bimodal* pooled static distribution can be aligned with a
*bimodal* pooled dynamic distribution by MMD. If dynamic were unimodal
(one-object) against a bimodal static, MMD would either smear or settle for a
non-zero floor.

## Inference (tac→latent) — which stats?

| Phase | Default behavior          | How to override                                |
|-------|---------------------------|------------------------------------------------|
| 1     | static stats from ckpt    | n/a — single stats block back then             |
| 2     | static stats from ckpt    | `--stats_dataset_name dataset_cross_modal_tri_dynamic_plug` |
| 3     | static stats from ckpt    | `--stats_dataset_name dataset_cross_modal_tri_dynamic_all`  |

The `--stats_dataset_name` flag in `convert_tac_to_latent_sachin.py` was added
during Phase 2/3. Without it, the converter reads
`dataset_details.yaml` from the checkpoint dir (always the static block) and
applies static stats to teleop data — fine for matching Phase-1 behavior, but
worse than feeding dynamic-trained encoders dynamic-normalized inputs.

**Always use the flag with Phase-2 and Phase-3 checkpoints when converting
teleop / dynamic-regime data.**

## Things considered and rejected

- **Per-object stats blocks inside `dataset_sachin`** (cyl gets cyl stats, plug
  gets plug stats during the same joint run). Rejected: erases cross-object
  magnitude information that representation learning should use.
- **One single global stats block pooled across static + dynamic.** Rejected:
  static frames would be under-normalized (std<1) and dynamic over-normalized
  (std>1) — MMD pays a penalty trying to bridge a pseudo-difference that's
  pure scale.
- **Cyl-only dynamic data on joint cyl+plug static.** Rejected: MMD has no way
  to match a unimodal dynamic distribution to a bimodal static distribution
  without distorting the encoder.

## Files of interest

- `config/datasets_config.yaml` — dynamic dataset entries.
- `config/datasets_config_sachin.yaml` — paired static dataset entry.
- `scripts/compute_per_object_stats.py` — produces `per_object_stats.yaml`.
- `joint_embedding_learning/data/datasets.py` — `get_dynamic_self_datasets`
  now accepts list `data_root` (Phase-3 change).
- `scripts/convert_tac_to_latent_sachin.py` — `--stats_dataset_name` flag for
  inference-time stats override.
- `scripts/train_dynamic_align_sachin.py` — training entry point.
