#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=llama2-pruning
#SBATCH --output=logs/slurm-%j.out

# Set common variables
model="meta-llama/Llama-2-7b-chat-hf"
sparsity_ratio=0.5
export CUDA_VISIBLE_DEVICES=1         # Start on a single available GPU, here GPU 1

# Set PyTorch memory management configuration to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Define function to run python command with logging, per template
run_python_command () {
    log_file="logs/$(basename $3)_$(date +'%Y%m%d_%H%M%S').log"
    mkdir -p $(dirname "$log_file")
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3 \
    &> "$log_file"
}

echo "Running pruning tasks with sparsity ratio $sparsity_ratio on a single GPU, sequentially to avoid memory overload."

# Run tasks in a conservative sequential manner
run_python_command "wanda" "unstructured" "out/llama_7b/unstructured/wanda/"
run_python_command "wanda" "2:4" "out/llama_7b/2-4/wanda/"
run_python_command "wanda" "4:8" "out/llama_7b/4-8/wanda/"

run_python_command "sparsegpt" "unstructured" "out/llama_7b/unstructured/sparsegpt/"
run_python_command "sparsegpt" "2:4" "out/llama_7b/2-4/sparsegpt/"
run_python_command "sparsegpt" "4:8" "out/llama_7b/4-8/sparsegpt/"

run_python_command "magnitude" "unstructured" "out/llama_7b/unstructured/magnitude/"
run_python_command "magnitude" "2:4" "out/llama_7b/2-4/magnitude/"
run_python_command "magnitude" "4:8" "out/llama_7b/4-8/magnitude/"

echo "All pruning tasks with sparsity 0.5 completed."
