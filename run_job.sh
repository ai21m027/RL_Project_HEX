#!/bin/sh
#SBATCH --job-name=hex-training-job
#SBATCH --output=/home/ai21m026/hex/slurm/hex-training-job_%A_%a.out
#SBATCH --gpus-per-task=1


. /opt/conda/etc/profile.d/conda.sh
conda activate venv

srun python CoachAssistant.py