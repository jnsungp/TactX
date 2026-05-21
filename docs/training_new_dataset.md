# dataset_sachin_new 학습 & 평가 가이드

`dataset_sachin_new`는 `tactile_benchmark/data/latent_data_new_processed`에 위치한 single-object (small_ball) 데이터셋으로, 총 150 에피소드 중 20개를 val로 사용하는 episode-level split을 적용한다.

## 데이터셋 요약


| 항목                  | 값                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------- |
| config 엔트리          | `dataset_sachin_new` (in `config/datasets_config_sachin.yaml`)                     |
| data_root           | `tactile_benchmark/data/latent_data_new_processed`                                 |
| object              | `small_ball`                                                                       |
| split 방식            | episode_val_split (20개 에피소드 holdout)                                               |
| val_episode_ids     | 0, 6, 13, 22, 36, 44, 45, 54, 58, 67, 75, 81, 88, 97, 111, 119, 120, 129, 133, 142 |
| stats               | small_ball 전용 (per_object_stats.yaml::small_ball_static)                           |
| pairs               | de (Daimon↔eFlesh), ef (eFlesh↔FlexiTac), fd (FlexiTac↔Daimon)                     |
| per-pair correction | rot180 (de: Daimon, ef: FlexiTac, fd: Daimon)                                      |


---

## 학습 (Training)

모든 학습 스크립트는 tri-modal paired-static 학습 (`train_tri`)을 사용하며, loss 구성만 다르다.
아래 예시는 `dataset_sachin_old_smallball`(옛날 데이터) 기준. 새 데이터를 쓰려면 `--dataset_name dataset_sachin_new`로 교체.

### 1. InfoNCE (Contrastive + Recon + KL) — 기본

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_static_sachin.py \
    --dataset_name dataset_sachin_new \
    --device cuda:0 \
    --run_name infonce_new \
    --objects small_ball \
    --epochs 300
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_static_sachin.py \
    --dataset_name dataset_sachin_old_smallball \
    --device cuda:0 \
    --run_name infonce_old_sb \
    --objects small_ball \
    --ef_stride 3 \
    --epochs 300
```


**Loss 구조** (`CrossModalVAE3Loss`):

```
Loss = α × Σ_p NT-Xent(μ_a, μ_b)           ← InfoNCE contrastive alignment
     + λ_recon × Σ_p (L1 recon, 4 branches)  ← self + cross reconstruction
     + β × Σ_p (KL_a + KL_b)                 ← KL divergence to N(0,I)
```


| 하이퍼파라미터              | 기본값         |
| -------------------- | ----------- |
| temperature          | 0.01        |
| lambda_recon         | 1.0         |
| beta_max             | 0.1         |
| kl_warmup_epochs     | 30          |
| infonce_warmup_start | 0 (처음부터 켜짐) |
| batch_size           | 64          |
| learning_rate        | 1e-4        |
| projection_dim       | 16          |


### 2. Uniforce / L2 (L2 Pull + Recon + KL)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_uniforce_sachin.py \
    --dataset_name dataset_sachin_new \
    --device cuda:0 \
    --run_name uniforce_new \
    --objects small_ball \
    --epochs 300
```

**Loss 구조** (`CrossModalVAE3UniforceLoss`):

```
Loss = α × Σ_p MSE(μ_a, μ_b)                ← L2 alignment (no negatives)
     + λ_recon × Σ_p (L1 recon, 4 branches)  ← self + cross reconstruction
     + β × Σ_p (KL_a + KL_b)                 ← KL divergence to N(0,I)
```

InfoNCE와의 차이: contrastive negative 없이 paired mean을 직접 L2로 당김.
collapse 방지는 reconstruction term에만 의존하므로 `lambda_recon`을 충분히 유지해야 한다.

### 3. Recon Only (Reconstruction + KL, alignment 없음)

InfoNCE alignment를 완전히 끄려면 `--infonce_warmup_start`를 총 epochs보다 크게 설정한다.
이렇게 하면 `_infonce_alpha` → 0으로 유지되어 alignment term이 전혀 적용되지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_static_sachin.py \
    --dataset_name dataset_sachin_new \
    --device cuda:0 \
    --run_name recon_only_new \
    --objects small_ball \
    --epochs 300 \
    --infonce_warmup_start 9999
```

**Loss 구조** (α=0인 `CrossModalVAE3Loss`):

```
Loss = λ_recon × Σ_p (L1 recon, 4 branches)  ← self + cross reconstruction
     + β × Σ_p (KL_a + KL_b)                 ← KL divergence to N(0,I)
```

각 modality encoder가 독립적으로 reconstruct하는 능력만 학습. cross-modal alignment 없음.

### 4. Contrastive Only (InfoNCE만, Recon/KL 없음)

별도의 모델 `cross_modal_clr3`를 사용한다. decoder가 없고, projection head + NT-Xent만 사용.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_static_sachin.py \
    --dataset_name dataset_sachin_new \
    --model_name cross_modal_clr3 \
    --device cuda:0 \
    --run_name clr3_new \
    --objects small_ball \
    --epochs 300
```

**Loss 구조** (`CrossModalCLR3Loss`):

```
Loss = Σ_p NT-Xent(z_a, z_b)   ← contrastive alignment only
```

No reconstruction, no KL. 순수 contrastive baseline.

### Loss 비교 요약


| 방식               | Alignment             | Reconstruction  | KL  | 스크립트                                                       | model_name         |
| ---------------- | --------------------- | --------------- | --- | ---------------------------------------------------------- | ------------------ |
| InfoNCE          | NT-Xent (contrastive) | L1 × 4 per pair | O   | `train_static_sachin.py`                                   | `cross_modal_vae3` |
| Uniforce/L2      | MSE (L2 pull)         | L1 × 4 per pair | O   | `train_uniforce_sachin.py`                                 | `cross_modal_vae3` |
| Recon Only       | X (α=0)               | L1 × 4 per pair | O   | `train_static_sachin.py` + `--infonce_warmup_start 9999`   | `cross_modal_vae3` |
| Contrastive Only | NT-Xent               | X               | X   | `train_static_sachin.py` + `--model_name cross_modal_clr3` | `cross_modal_clr3` |


### 공통 옵션


| 옵션                        | 설명                              | 기본값              |
| ------------------------- | ------------------------------- | ---------------- |
| `--dataset_name`          | datasets_config_sachin.yaml 엔트리 | `dataset_sachin` |
| `--objects`               | 학습할 오브젝트                        | `cylinder plug`  |
| `--epochs`                | 총 에폭 수                          | 100 (config)     |
| `--ef_stride`             | ef_pair 프레임 서브샘플링 stride        | 1                |
| `--de_stride`             | de_pair 프레임 서브샘플링 stride        | 1                |
| `--fd_stride`             | fd_pair 프레임 서브샘플링 stride        | 1                |
| `--projection_dim`        | latent 차원 override              | 16 (config)      |
| `--lr`                    | learning rate override          | 1e-4 (config)    |
| `--infonce_warmup_start`  | alignment term 활성화 시작 epoch     | 0                |
| `--infonce_warmup_epochs` | alignment weight 0→1 ramp 폭     | 0                |
| `--resume_from`           | 체크포인트에서 weight warm-start       | None             |


---

## 평가 (Evaluation)

`scripts/eval_cross_modal_vae3_new.py`를 사용한다.
학습 때와 동일한 `dataset_name` + `objects` + stride를 넘겨야 val split이 일치한다.

### Static만 평가

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cross_modal_vae3_new.py \
    --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/<RUN_DIR> \
    --dataset_name dataset_sachin_new \
    --objects small_ball \
    --ef_stride 3 \
    --device cuda:0
```



### Static + Dynamic 평가

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cross_modal_vae3_new.py \
    --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/<RUN_DIR> \
    --dataset_name dataset_sachin_new \
    --objects small_ball \
    --ef_stride 3 \
    --dynamic_data_root latent_data_processed_dynamic/pnp latent_data_processed_dynamic/wipe \
    --dynamic_objects pnp wipe \
    --device cuda:0
```

### 출력물 (`<checkpoint_dir>/eval_new/`)


| 파일                           | 내용                                                  |
| ---------------------------- | --------------------------------------------------- |
| `tsne_static_train.png`      | static train set t-SNE (modality 색상 + pair link)    |
| `tsne_static_val.png`        | static val set t-SNE                                |
| `cosine_dist_val.png`        | DE/EF/FD/DF(transitive) cosine gap 히스토그램            |
| `recon_daimon.png`           | Daimon reconstruction 시각화 (GT / self / cross)       |
| `recon_eflesh.png`           | eFlesh reconstruction 시각화                           |
| `recon_flexi.png`            | FlexiTac reconstruction 시각화                         |
| `metrics.txt`                | cosine gap, retrieval top-k, L1 recon error, MMD 수치 |
| `tsne_dynamic_paired.png`    | dynamic t-SNE (--dynamic_data_root 시)               |
| `tsne_dynamic_bytask.png`    | dynamic task별 색상 (multi-root 시)                     |
| `tsne_static_vs_dynamic.png` | static vs dynamic 오버레이                              |


### 실제 예시 (infonce_new 체크포인트)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cross_modal_vae3_new.py \
    --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_sachin_new_run_infonce_new_efs3_20260520_074504 \
    --dataset_name dataset_sachin_new \
    --objects small_ball \
    --ef_stride 3 \
    --dynamic_data_root latent_data_processed_dynamic/pnp latent_data_processed_dynamic/wipe \
    --dynamic_objects pnp wipe \
    --device cuda:0
```

### eval 옵션


| 옵션                                            | 설명                                | 기본값                         |
| --------------------------------------------- | --------------------------------- | --------------------------- |
| `--checkpoint_dir`                            | 체크포인트 디렉토리 (model.pt 포함)          | 필수                          |
| `--dataset_name`                              | 학습 때 사용한 dataset 엔트리              | `dataset_sachin_new`        |
| `--objects`                                   | 학습 때 사용한 오브젝트                     | `small_ball`                |
| `--ef_stride` / `--de_stride` / `--fd_stride` | 학습 때 사용한 stride (val split 일치 보장) | 1                           |
| `--dynamic_data_root`                         | dynamic 데이터 경로 (없으면 스킵)           | None                        |
| `--dynamic_objects`                           | dynamic_data_root와 1:1 대응 오브젝트명   | None                        |
| `--out_dir`                                   | 출력 디렉토리 override                  | `<checkpoint_dir>/eval_new` |
| `--max_tsne_points`                           | t-SNE 포인트 상한 (0=전체)               | 0                           |
| `--tsne_perplexity`                           | t-SNE perplexity override         | auto                        |


