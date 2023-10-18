#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=dinov99
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20g
#SBATCH --mail-user=tingtind@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=01-00:00:00
module purge
module load gcc cuda/11.7.1 cudnn/11.7-v8.7.0 python
source env/bin/activate


export 'HUGGINGFACE_TOKEN=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'

huggingface-cli login --token $HUGGINGFACE_TOKEN

python benchmark_medmcqa.py
