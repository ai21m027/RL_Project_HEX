#!/bin/sh

rm logs/*
rm slurm/hex-training-job*

sbatch ./run_training.sh

echo "use squeue to view running jobs"