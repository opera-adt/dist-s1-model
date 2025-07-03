#!/bin/bash
#SBATCH -J train_opera           # Job name
#SBATCH -o logs/tunnel_%j.o           # Stdout (%j = job ID)
#SBATCH -e logs/tunnel_%j.e           # Stderr (%j = job ID)
#SBATCH -p gpu            # Queue (partition) name
#SBATCH -G 1
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 32            # Total # of mpi tasks
#SBATCH --mem=128G            # Memory (RAM) requested
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=john.mauro@jpl.nasa.gov
#SBATCH --ntasks-per-core=1
#SBATCH --account=opera-dist-ml 

### Load modules into your environment
module load netcdf/impi/intel/4.9.2
source /cm/shared/apps/intel/oneapi/setvars.sh
module load conda
source /cm/shared/apps/conda/etc/profile.d/conda.sh
conda activate /scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu


### Run
#cd /scratch-jpl/opera-dist-ml/users/jmauro/dist-s1-model/
#python trainer.py

cd /scratch-jpl/opera-dist-ml/users/jmauro/dist-s1-model/

/scratch-jpl/opera-dist-ml/users/jmauro/envs/dist-s1-model-gpu/bin/python trainer.py
