#!/bin/bash
#SBATCH -J create_data
#SBATCH -o ./logs/%j.out

python -m src.create_data