# DIST-S1 Model

This is a repository that includes the transformer model and relevant training routines.
It is a greatly distilled version of Harris Hardiman-Mostow's research [repository](https://github.com/OPERA-Cal-Val/deep-dist-s1-research) with optimizations and improvements specifically tailored for the DIST-S1 product written by [Diego Martinez](https://github.com/dmartinez05). There are also additional notebooks to inspect the input dataset and visualize the model application to existing OPERA RTC data.

## Installation

### Environment Setup

1. Install the environment using mamba:
   ```bash
   mamba env create -f environment_gpu.yml
   ```

2. Activate the environment:
   ```bash
   conda activate dist-s1-model
   ```

## Data Setup

### Download Required Datasets

1. **Training data** (~53 GB): `<url>`
2. **Test data** (~13 GB): `<url>`

### Data Configuration

Update the data paths in your configuration file (see Configuration section below).

## Usage

**Note:**

We currently support three different style of datasets:

1. `v0`: sequential time-series to establish baselines that have been denoised, and nodata and surface water filled in with 0s
2. `v1`: Same pre-processing as 1. except curated around anniversary date from the target/post-image acquisition to establish a baseline.
3. `v2`: No pre-processing (i.e. no despeckling and no mask filling) using the curation from 2. - though additionally allow variable time-lengths 

4. is the original data curation that was done to demonstrate this approach in Hardiman, et al. 2. represents what OPERA project aims to support to be in line with the OPERA DIST suite. 3. is to further push the ability of the model in order to perform both despeckling and baseline estimation.
Currently all the `*-redux` or `Redux` are for 3.

### Downloading data

1. `v0` can be downloaded from this public `s3` bucket: `s3://opera-dist-s1-training-data/v0` (~60 GB)
2. `v1` can be downloaded from this public `s3` bucket: `s3://opera-dist-s1-training-data/v1` (~75 GB)
3. `v2` can be generated from this repository (note it is approximately 27 TB): https://github.com/opera-adt/dist-s1-training-data

### YAML Configuration File

Create a configuration file (e.g., `config.yml`) with the following structure:

```yaml
# Data configuration
data:
  train_path: "/path/to/your/train_data.pt"
  test_path: "/path/to/your/test_data.pt"

# Model configuration
model_config:
  type: "SpatioTemporalTransformer"
  # Add your model-specific parameters here

# Training configuration
train_config:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 100
  seed: 42
  step_size: 30
  gamma: 0.1
  checkpoint_freq: 10
  input_size: 16  # Patch size for processing

# Save directories
save_dir:
  models: "./saved_models"
  checkpoints: "./checkpoints"
  visualizations: "./visualizations"

# Validation configuration (optional)
validation:
  enable_visual_validation: true
  enable_intermediate_validation: true
  intermediate_validation_freq: 10
  apply_smoothing: true
  smooth_sigma: 0.5
  blend_mode: "gaussian"

# Weights & Biases logging (optional)
use_wandb: true
wandb_project: "dist-s1-training"
wandb_entity: "your-entity"

# Resume training (optional)
# resume_checkpoint: "/path/to/checkpoint.pth"
```

### Accelerate Configuration

#### Option 1: Interactive Configuration

Set up Accelerate configuration interactively:

```bash
accelerate config
```

Follow the prompts to configure:
- Compute environment (local machine or cluster)
- Machine type (multi-GPU, multi-node, etc.)
- Number of processes/GPUs
- Mixed precision settings

#### Option 2: Manual Configuration

Create an Accelerate config file (`accelerate_config.yml`):

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU  # or NO for single GPU
gpu_ids: all  # or specify specific GPUs like "0,1"
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 2  # Number of GPUs to use
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Training

### Single GPU Training

```bash
python trainer.py config.yml
```
or

```bash
python trainer_redux.py config_redux.yml
```

### Multi-GPU Training with Accelerate

#### Using Default Accelerate Config

```bash
accelerate launch trainer.py config.yml
```

#### Using Custom Accelerate Config

```bash
accelerate launch --config_file accelerate_config.yml train.py config.yml
```

#### Direct Launch with Parameters

```bash
accelerate launch --num_processes 2 train.py config.yml
```

### Advanced Training Options

#### Disable Torch Compilation

If you encounter issues with PyTorch's dynamo compilation, you can disable it by setting the environment variable:

```bash
export TORCH_COMPILE_DISABLE=1
accelerate launch train.py config.yml
```

#### Resume Training from Checkpoint

Add the checkpoint path to your config:

```yaml
resume_checkpoint: "/path/to/checkpoint_epoch_X.pth"
```

#### Preserve Standard I/O

To capture training logs:

```bash
accelerate launch train.py config.yml > training.log 2> training.err
```

## Monitoring and Validation

### Weights & Biases Integration

The training script supports Weights & Biases logging. Configure in your YAML:

```yaml
use_wandb: true
wandb_project: "your-project-name"
wandb_entity: "your-entity"
```

### Wandb Setup

Before using wandb for the first time, you must open a terminal session, activate the `dist-s1-model` env and run
`wandb login`. The command line will prompt you for an API key that can be found at https://wandb.ai/home.

### Visual Validation

Enable visual validation to monitor training progress:

```yaml
validation:
  enable_visual_validation: true
  enable_intermediate_validation: true
  intermediate_validation_freq: 10
```

### Checkpointing

Checkpoints are automatically saved based on the `checkpoint_freq` setting. The training script creates:

- Regular checkpoints: `checkpoint_epoch_X_MM-DD-YYYY_HH-MM.pth`
- Model weights: `ModelType_MM-DD-YYYY_HH-MM_epoch_X.pth`
- Final checkpoint: `final_checkpoint_MM-DD-YYYY_HH-MM.pth`
- Emergency checkpoints: Saved automatically on interruption

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` in your configuration
2. **Compilation Errors**: Set environment variable `TORCH_COMPILE_DISABLE=1`
3. **Multi-GPU Issues**: Ensure proper Accelerate configuration
4. **Data Loading Errors**: Verify data paths in configuration file

### Performance Tips

- Adjust `input_size` based on available GPU memory
- Enable gradient accumulation in Accelerate config for larger effective batch sizes

### Graceful Interruption

The training script supports graceful interruption (Ctrl+C). It will:
- Save an emergency checkpoint
- Preserve training metrics
- Clean up resources properly

## Application

See the included notebooks for model application examples. This section is currently under development.

## Data Curation

A separate repository for SAR data curation is planned. This is currently a work in progress.

## How to Use The Benchmark Script

The [benchmark script](benchmark.py) is designed to evaluate multiple models against multiple datasets. To use it, you need to correctly place your model files and configure the dataset paths within the script itself.

---

### 1. Adding Your Models

The script automatically discovers models by scanning a specific folder, but requires a strict naming convention.

- **Create a folder** for your models. By default, the script looks for a folder named `model_data`. You can change this by modifying the `MODELS_DIR` variable.
- For each model you want to test, place two files inside this folder, named as follows:
    1. A **`.yml` file** for configuration, named **`config_YOUR_MODEL_NAME.yml`**.
    2. A **`.pth` file** for weights, named **`checkpoint_YOUR_MODEL_NAME.pth`**.
- **Important:** The `YOUR_MODEL_NAME` part must be identical between the two files for them to be paired correctly.

**Example Directory Structure:**

    .
    ├── benchmark_script.py
    └── model_data/
        ├── config_transformer_small.yml
        ├── checkpoint_transformer_small.pth
        ├── config_transformer_large.yml
        └── checkpoint_transformer_large.pth

---

### 2. Configuring Your Datasets

You must define the datasets you want to evaluate directly within the Python script.

- **Locate the `DATASETS_TO_TEST` list** in the main section of the script.
- **Add a dictionary** to this list for each dataset you want to test.

Each dataset dictionary requires the following keys:

- `"type"`: The loader type. Use `"dataset_v0"` for standard `.pt` files loaded with `torch.load` or `"dataset_v1"` for your `StreamShardDataset`.
- `"name"`: A short, descriptive name for the dataset (e.g., "ERA5 Hourly"). This name will appear in the final report.
- `"train_path"`: The full path to your training data file or directory.
- `"test_path"`: The full path to your testing data file or directory.
- `"seq_len"`: The native sequence length of the data in this dataset.

**Example Configuration:**

    # Define the datasets you want to test each model against.
    DATASETS_TO_TEST = [
        {
            "type": "dataset_v0",
            "name": "V0",
            "train_path": "PytorchData/train_12813.pt",
            "test_path": "Pytor/chData/test_3204.pt",
            "seq_len": 10
        },
        {
            "type": "dataset_v1",
            "name": "V1",
            "train_path": "opera-dist-ml/data/v1/",
            "test_path": "opera-dist-ml/data/v1/",
            "seq_len": 20
        }
    ]


## References

- OPERA Disturbance Suite: https://www.jpl.nasa.gov/go/opera/products/dist-product-suite/

- Hardiman-Mostow, Harris, Charles Marshak, and Alexander L. Handwerger. "Deep Self-Supervised Disturbance Mapping with the OPERA Sentinel-1 Radiometric Terrain Corrected SAR Backscatter Product." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2025). [arXiv](https://arxiv.org/abs/2501.09129)


## Contributing

This is an open-source research repository to provide provenance of the models used to measure disturbance.
Please open up an issue and we can work together to fix bugs.

## Support

For issues and questions, please create an issue in this repository or contact the maintainers.
