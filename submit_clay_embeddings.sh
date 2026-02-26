#!/bin/bash
#SBATCH -J clay_embed              # Job name
#SBATCH -o logs/clay_embed.o%j     # Stdout log file (%j = job ID)
#SBATCH -e logs/clay_embed.e%j     # Stderr log file
#SBATCH -p gpu                     # Queue (partition) name
#SBATCH -G 1                       # 1 GPU
#SBATCH -N 1                       # 1 node
#SBATCH -n 8                       # 8 CPU tasks (for data loading)
#SBATCH --mem=64G                  # 64GB RAM (Clay is ~5GB, rest for data I/O)
#SBATCH -t 72:00:00                # 72 hours wall time
#SBATCH --mail-type=all
#SBATCH --mail-user=john.mauro@jpl.nasa.gov
#SBATCH --ntasks-per-core=1
#SBATCH --account=opera-dist-ml

### Load modules
module load netcdf/impi/intel/4.9.2
source /cm/shared/apps/intel/oneapi/setvars.sh
module load conda-miniforge/24.3.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/jmauro/.conda/envs/dist-s1-model

### Fix GLIBCXX for Clay/Lightning
export LD_LIBRARY_PATH=/home/jmauro/.conda/envs/dist-s1-model/lib:$LD_LIBRARY_PATH

### Change to project directory
cd /scratch-jpl/opera-dist-ml/users/jmauro/dist-s1-model

### Ensure output dirs exist
mkdir -p logs
mkdir -p clay_embeddings

### Run Clay embedding computation
/home/jmauro/.conda/envs/dist-s1-model/bin/python compute_clay_embeddings.py \
    --checkpoint clay-v1.5.ckpt \
    --metadata-path configs/metadata.yaml \
    --data-dir /scratch/opera-dist-ml/users/jmauro/dist-s1-model \
    --output-dir clay_embeddings \
    --batch-size 16
