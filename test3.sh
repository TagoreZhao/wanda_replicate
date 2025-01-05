#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=llama2-pruning
#SBATCH --output=logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4    # Allocate 4 CPU cores per GPU task

# Set common variables
model="meta-llama/Llama-2-7b-hf"
sparsity_ratio=0.5

# Set CUDA device visibility to use all available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the pruning script with torchrun across 4 GPUs
echo "Running with wanda pruning method across 4 GPUs"

torchrun --nproc_per_node=4 main.py \
    --model $model \
    --prune_method "wanda" \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "unstructured" \
    --save "out/llama_7b/unstructured/wanda/" \
    --eval_zero_shot

echo "Finished wanda pruning"
# Explicitly exit
exit 0
