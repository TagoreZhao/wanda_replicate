#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=llama2-pruning
#SBATCH --output=logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4  # Total tasks (one per GPU)
#SBATCH --cpus-per-task=4  # Allocate 4 CPU cores per GPU

# Load necessary modules
module load mpi/openmpi-x86_64

# Set common variables
model="meta-llama/Llama-2-7b-chat-hf"
sparsity_ratio=0.5
export MASTER_PORT=$((12000 + RANDOM % 1000))  # Unique port for each job
export OMP_NUM_THREADS=1
export RANK=$SLURM_PROCID
# Make all 4 GPUs visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Define function to run the Python command
run_python_command () {
    torchrun --nproc_per_node=4 \
    --rdzv_conf join_timeout=1000000 \
    --max_restarts 10 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3
}

# Create output directories if not exist
mkdir -p logs
mkdir -p out/llama_7b/{unstructured,2-4,4-8}/{wanda,sparsegpt,magnitude}

# Run pruning with different methods
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "out/llama_7b/unstructured/wanda/"
run_python_command "wanda" "2:4" "out/llama_7b/2-4/wanda/"
run_python_command "wanda" "4:8" "out/llama_7b/4-8/wanda/"
echo "Finished wanda pruning method"

echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/llama_7b/unstructured/sparsegpt/"
run_python_command "sparsegpt" "2:4" "out/llama_7b/2-4/sparsegpt/"
run_python_command "sparsegpt" "4:8" "out/llama_7b/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"

echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/llama_7b/unstructured/magnitude/"
run_python_command "magnitude" "2:4" "out/llama_7b/2-4/magnitude/"
run_python_command "magnitude" "4:8" "out/llama_7b/4-8/magnitude/"
echo "Finished magnitude pruning method"
