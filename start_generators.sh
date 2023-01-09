#!/bin/sh

#SBATCH --job-name=hex-generator
#SBATCH --output=/home/ai21m026/hex/slurm/hex-generator-%A_%a.out
#SBATCH --array=1-100

. /opt/conda/etc/profile.d/conda.sh
conda activate generator_venv

srun python -O DataGenerator.py