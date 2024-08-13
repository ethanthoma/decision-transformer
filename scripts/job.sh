#!/usr/bin/env bash

#SBATCH --account=st-gerope-1-gpu
#SBATCH --time=4:0:0
#SBATCH --ntasks=1
#SBATCH --nodes=01
#SBATCH --output=output/%j.out
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=DT
#SBATCH --constraint=gpu_mem_32

lscpu
nvidia-smi

./scripts/load_modules.sh

source decision-transformer/bin/activate

./scripts/link_cuda.sh

# run eval
CUDA=1 python -m app.main train
