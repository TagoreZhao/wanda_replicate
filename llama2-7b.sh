#!/bin/bash

# Set common variables
model="baffo32/decapoda-research-llama-7B-hf"
sparsity_ratio=0.5

# Function to check available GPUs (idle or low memory usage)
get_available_gpus() {
  available_gpus=()
  for gpu_id in {0..3}; do
    # Check memory usage for each GPU (using nvidia-smi)
    memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    if [ "$memory_used" -lt 1000 ]; then  # Consider GPUs with less than 1GB used as available
      available_gpus+=($gpu_id)
    fi
  done
  echo "${available_gpus[@]}"
}

# Get list of available GPUs
cuda_devices=$(get_available_gpus)

# Check if we have any available GPUs
if [ -z "$cuda_devices" ]; then
  echo "No available GPUs found. Exiting."
  exit 1
else
  echo "Available GPUs: $cuda_devices"
fi

# Set CUDA device visibility for available GPUs
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${cuda_devices[*]}")

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
