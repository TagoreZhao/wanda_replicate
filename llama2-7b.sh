#!/bin/bash

# Set common variables
model="meta-llama/Llama-2-7b-chat-hf"
sparsity_ratio=0.5
cuda_devices=(0 1 2 3)  # Use all four GPUs if available
precision="fp16"  # Mixed precision for optimized memory usage

# Set CUDA device visibility for multi-GPU use
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${cuda_devices[*]}")

# Define function to run python command with logging
run_python_command () {
   log_file="logs/$(basename $3)_$(date +'%Y%m%d_%H%M%S').log"
   mkdir -p $(dirname "$log_file")
   python main.py \
   --model $model \
   --prune_method $1 \
   --sparsity_ratio $sparsity_ratio \
   --sparsity_type $2 \
   --save $3 \
   --precision $precision \
   --multi_gpu True \
   &> "$log_file"
}

echo "Running pruning tasks with sparsity ratio $sparsity_ratio"

# wanda pruning method
echo "Running wanda pruning method with sparsity 0.5"
run_python_command "wanda" "unstructured" "out/llama_7b/unstructured/wanda/" &
run_python_command "wanda" "2:4" "out/llama_7b/2-4/wanda/" &
run_python_command "wanda" "4:8" "out/llama_7b/4-8/wanda/" &

# sparsegpt pruning method
echo "Running sparsegpt pruning method with sparsity 0.5"
run_python_command "sparsegpt" "unstructured" "out/llama_7b/unstructured/sparsegpt/" &
run_python_command "sparsegpt" "2:4" "out/llama_7b/2-4/sparsegpt/" &
run_python_command "sparsegpt" "4:8" "out/llama_7b/4-8/sparsegpt/" &

# magnitude pruning method
echo "Running magnitude pruning method with sparsity 0.5"
run_python_command "magnitude" "unstructured" "out/llama_7b/unstructured/magnitude/" &
run_python_command "magnitude" "2:4" "out/llama_7b/2-4/magnitude/" &
run_python_command "magnitude" "4:8" "out/llama_7b/4-8/magnitude/" &

wait  # Wait for all background processes to finish

echo "All pruning tasks with sparsity 0.5 completed."
