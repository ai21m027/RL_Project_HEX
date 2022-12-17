#!/bin/sh
#SBATCH --job-name=hex-training
#SBATCH --output=/home/ai21m026/hex/slurm/hex-training.out

. /opt/conda/etc/profile.d/conda.sh
conda activate venv

python main_train.py