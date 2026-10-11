# Downstream ACT policy workflow

TactX policies consume frozen tactile encoder outputs alongside camera observations and robot state. This guide follows the 16-D latent path implemented in this repository. It assumes that paired representation training has already produced a `cross_modal_vae3` checkpoint and that processed robot demonstrations are available locally.

## 1. Convert tactile observations to latents

Use the Python 3.10 `tactx` environment from the [root README](../README.md#installation), and run from the repository root. Set the checkpoint path and input/output directories for your own episodes:

```bash
conda activate tactx
CHECKPOINT_DIR="/path/to/trained/run"

CUDA_VISIBLE_DEVICES=0 python -m scripts.convert_tac_to_latent_sachin_updated \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_file model.pt \
    --sensor F \
    --input_dir /path/to/flexitac_train_episodes \
    --output_dir tactile_benchmark/data/recordings/processed/TACTX/flexi_flexi_object_pnp_latent_TACTX \
    --latent_type mu \
    --device cuda:0

CUDA_VISIBLE_DEVICES=0 python -m scripts.convert_tac_to_latent_sachin_updated \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_file model.pt \
    --sensor F \
    --input_dir /path/to/flexitac_val_episodes \
    --output_dir tactile_benchmark/data/recordings/processed/TACTX/flexi_flexi_object_pnp_val_latent_TACTX \
    --latent_type mu \
    --device cuda:0
```

The converter reads `episode_*.hdf5` files and uses `model_details.yaml` and `dataset_details.yaml` from the checkpoint directory. It replaces `observation.tac0` and `observation.tac1` with posterior means, copies the other root-level datasets, and preserves file/dataset attributes. Use distinct input and output directories.

Choose the encoder with `--sensor`:

| Flag | Sensor | Expected input in each `observation.tac*` dataset |
| :--- | :--- | :--- |
| `D` | Daimon | `(T, 384000)`: flattened `240 × 320 × 5` observations; the converter selects channels `[0, 3, 4]` |
| `E` | eFlesh | `(T, 15)` |
| `F` | FlexiTac | `(T, 192)`: flattened `12 × 16` grids |

With a 16-D checkpoint, each output tactile stream has shape `(T, 16)`. Daimon conversion uses the raw depth/shear data, rather than the RGB tactile visualization. The chosen encoder is applied to both fingers; this command assumes that both tactile streams use the selected sensor type.

The converter defaults to no geometric correction at inference. Set `--inference_transform` to `fliplr`, `flipud`, or `rot180` only when needed to match the orientation used to train the checkpoint. It uses the saved normalization statistics unless `--stats_dataset_name` explicitly selects another entry from `config/datasets_config.yaml`.

## 2. Set up the policy environment

The [policy package](pyproject.toml) requires Python 3.11. From the repository root:

```bash
conda create -n tactx_policy python=3.11 -y
conda activate tactx_policy
python -m pip install -r requirements.txt -r tactile_benchmark/requirements.txt
python -m pip install -e ./tactile_benchmark
```

The root requirements supply PyTorch, torchvision, NumPy, and scikit-learn; the policy requirements add Accelerate, OmegaConf, and the remaining policy dependencies. Robot drivers and sensor hardware setup are separate from this offline training environment.

## 3. Check the demonstration layout

The included [pnp_flexi_latent_TACTX.json](tactile_policy/configs/datasets/pnp_flexi_latent_TACTX.json) selects these folders relative to `tactile_benchmark/data/recordings/processed/`:

```text
TACTX/
├── flexi_flexi_object_pnp_latent_TACTX/
│   ├── episode_0000.hdf5
│   └── ...
└── flexi_flexi_object_pnp_val_latent_TACTX/
    ├── episode_0000.hdf5
    └── ...
```

For another task or sensor, edit or create a JSON split file in `tactile_benchmark/tactile_policy/configs/datasets/` with `train` and `val` lists of folder names. The repository also includes `*_latent_TACTX.json` splits for other sensor/task combinations.

The ACT configuration below uses padded 128-D `observation.state` and `action`, camera streams `observation.image.cam0` and `observation.image.cam1`, and 16-D `observation.tac0` and `observation.tac1`. Camera frames have shape `(T, 240, 320, 3)`. These dotted names are root-level HDF5 dataset names. Keep the demonstration metadata, including any `embodiment` attribute, when preparing episodes.

## 4. Train an ACT policy

Run from `tactile_benchmark/`, because policy configuration paths are resolved relative to that directory:

```bash
cd tactile_benchmark

CUDA_VISIBLE_DEVICES=0 python -m tactile_policy.main \
    --batch_size 8 \
    --num_epochs 50000 \
    --lr 1e-5 \
    --chunk_size 64 \
    --exptid pnp_flexi_tactx \
    --dataset_json pnp_flexi_latent_TACTX \
    --model_cfg act_cross_modal_pnp_z16 \
    --no_wandb
```

In this trainer, `--num_epochs` controls the training-iteration limit despite its name. `--no_wandb` disables experiment logging; omit it to log online.

The [model configuration](tactile_policy/configs/models/act_cross_modal_pnp_z16.yaml) selects ACT with two cameras and two tactile streams. Its [sensor configuration](tactile_policy/configs/sensors/cross_modal_latent_z16.yaml) specifies a 16-D array input and an MLP tactile encoder. Policy weights and normalization statistics are written to `tactile_benchmark/data/checkpoints/pnp_flexi_tactx_ckpt/`.

## 5. Transfer across sensors

At deployment, use the target sensor's TactX encoder and the checkpoint-compatible preprocessing to supply the same latent interface to the trained policy. Robot execution additionally requires the matching camera/state inputs, action normalization, robot controller, and sensor setup. Offline validation on recorded demonstrations does not measure the paper's real-robot task success rates.
