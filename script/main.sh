#!/bin/bash
#SBATCH -J finetune
#SBATCH -o ./logs/%j/slurm.out

# python -m src.main --job_id $SLURM_JOB_ID --use_custom_loss
singularity run --nv singularity.sif \
    poetry run \
    python -m src.main --job_id $SLURM_JOB_ID --use_custom_loss
