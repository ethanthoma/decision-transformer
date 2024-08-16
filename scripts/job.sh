#!/usr/bin/env bash

#SBATCH --account=st-gerope-1-gpu
#SBATCH --time=5:30:0
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

source scripts/load_modules.sh
source scripts/link_cuda.sh

source decision-transformer/bin/activate

pip list

# run eval
mkdir -p models
CUDA=1 python -m app.main train
