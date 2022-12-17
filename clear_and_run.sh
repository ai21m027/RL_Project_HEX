#!/bin/sh

rm slurm/*

sbatch ./run_training.sh

echo "use squeue to view running jobs"