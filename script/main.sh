#!/bin/bash
#SBATCH -J finetune
#SBATCH -o ./logs/%j/slurm.out

singularity run --nv singularity.sif \
    poetry run \
    python -m src.main --job_id $SLURM_JOB_ID
