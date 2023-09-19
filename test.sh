#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=dinov99
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20g
#SBATCH --mail-user=tingtind@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
module purge
module load gcc cuda/11.7.1 cudnn/11.7-v8.7.0 python
# source /nfs/turbo/umms-dinov/LLaMA/2.0.0/bin/activate
source env/bin/activate

# llama2 7b text completion model
# torchrun --nproc_per_node 1 /home/tingtind/llama/example_text_completion.py \
#     --ckpt_dir /nfs/turbo/umms-dinov/LLaMA/2.0.0/llama/modeltoken/llama-2-7b \
#     --tokenizer_path /nfs/turbo/umms-dinov/LLaMA/1.0.1/llama/modeltoken/tokenizer.model

python finetune_peft.py
