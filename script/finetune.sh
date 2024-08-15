#!/bin/bash
#SBATCH -J finetune
#SBATCH -o ./logs/%j/slurm.out

python -m src.finetune --job_id $SLURM_JOB_ID 