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

## Configuration

### YAML Configuration File

Create a configuration file (e.g., `config.yaml`) with the following structure:

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

Create an Accelerate config file (`accelerate_config.yaml`):

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
python train.py config.yaml
```

### Multi-GPU Training with Accelerate

#### Using Default Accelerate Config

```bash
accelerate launch train.py config.yaml
```

#### Using Custom Accelerate Config

```bash
accelerate launch --config_file accelerate_config.yaml train.py config.yaml
```

#### Direct Launch with Parameters

```bash
accelerate launch --num_processes 2 train.py config.yaml
```

### Advanced Training Options

#### Disable Torch Compilation

If you encounter issues with PyTorch's dynamo compilation, you can disable it by setting the environment variable:

```bash
export TORCH_COMPILE_DISABLE=1
accelerate launch train.py config.yaml
```

#### Resume Training from Checkpoint

Add the checkpoint path to your config:

```yaml
resume_checkpoint: "/path/to/checkpoint_epoch_X.pth"
```

#### Preserve Standard I/O

To capture training logs:

```bash
accelerate launch train.py config.yaml > training.log 2> training.err
```

## Monitoring and Validation

### Weights & Biases Integration

The training script supports Weights & Biases logging. Configure in your YAML:

```yaml
use_wandb: true
wandb_project: "your-project-name"
wandb_entity: "your-entity"
```

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

## References

- OPERA Disturbance Suite: https://www.jpl.nasa.gov/go/opera/products/dist-product-suite/

- Hardiman-Mostow, Harris, Charles Marshak, and Alexander L. Handwerger. "Deep Self-Supervised Disturbance Mapping with the OPERA Sentinel-1 Radiometric Terrain Corrected SAR Backscatter Product." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2025). [arXiv](https://arxiv.org/abs/2501.09129)

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]

## Support

For issues and questions, please create an issue in this repository or contact the maintainers.