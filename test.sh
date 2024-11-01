#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=llama2-pruning
#SBATCH --output=logs/slurm-%j.out
#SBATCH --ntasks=4
#SBATCH --nodes=1

# Set common variables
model="meta-llama/Llama-2-7b-chat-hf"
sparsity_ratio=0.5
export MASTER_PORT=$((12000 + RANDOM % 1000))  # Unique port for each job
export OMP_NUM_THREADS=1

# Run pruning with torchrun
echo "Running wanda pruning with MPI on a single node with 4 GPUs"

mpirun -np 4 torchrun \
    --nproc_per_node 4 \
    --rdzv_conf join_timeout = 1000 \
    --max_restarts 3 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:${MASTER_PORT} \
    main.py \
    --model $model \
    --prune_method "wanda" \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "2:4" \
    --save "out/llama_7b/2-4/wanda/"

wait

echo "Finished wanda pruning with MPI on a single node"
# Explicitly exit
exit 0
