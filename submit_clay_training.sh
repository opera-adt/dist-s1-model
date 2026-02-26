#!/bin/bash
#SBATCH -J clay_train            # Job name
#SBATCH -o logs/clay_train.o%j        # Stdout log file (%j = job ID)
#SBATCH -e logs/clay_train.e%j        # Stderr log file
#SBATCH -p gpu                   # Queue (partition) name
#SBATCH -G 1
#SBATCH -N 1                     # Total # of nodes
#SBATCH -n 32                    # Total # of mpi tasks
#SBATCH --mem=128G              # Memory (RAM) requested
#SBATCH -t 72:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --mail-user=john.mauro@jpl.nasa.gov
#SBATCH --ntasks-per-core=1
#SBATCH --account=opera-dist-ml

### Load modules into your environment
module load netcdf/impi/intel/4.9.2
source /cm/shared/apps/intel/oneapi/setvars.sh

# Load the Miniforge Conda module
module load conda-miniforge/24.3.0

# Manually source conda's shell function support
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your conda environment
conda activate /scratch/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu

### Change to project directory
cd /scratch-jpl/opera-dist-ml/users/jmauro/dist-s1-model

### Ensure output directories exist
mkdir -p logs
mkdir -p models_clay
mkdir -p checkpoints_clay

### Run the Clay temporal model training script
/scratch/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/bin/accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    trainer_clay.py config_clay.yml
