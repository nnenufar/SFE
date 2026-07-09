#!/bin/bash
#SBATCH --job-name=sfe
#SBATCH --output=slurm/out/sfe.out
#SBATCH --error=slurm/out/sfe.err
#SBATCH --time=02-00
#SBATCH --mem=10G

export HOME=/home/$USER
export IN_DIR=<path>
export OUT_DIR=<path>

source ~/miniconda3/bin/activate
conda activate sfe

python src/main.py \
-in $IN_DIR \
-out $OUT_DIR \
-env bark \
-sr 16000
