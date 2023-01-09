#!/bin/sh

#SBATCH --job-name=hex-trainer
#SBATCH --output=/home/ai21m026/hex/slurm/hex-trainer-%A_%a.out
#SBATCH --array=1

. /opt/conda/etc/profile.d/conda.sh
conda activate venv

srun python -O ModelTrainer.py