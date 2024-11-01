#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=llama2-pruning
#SBATCH --output=logs/slurm-%j.out
#SBATCH --nodes=1            # Ensure we’re using only one node
#SBATCH --ntasks=4           # Set the number of MPI tasks to 4

# Set common variables
model="meta-llama/Llama-2-7b-chat-hf"
sparsity_ratio=0.5

# Launch the script with mpirun on a single node with 4 tasks
echo "Running with MPI and wanda pruning method on a single node with 4 GPUs"

# Use mpirun to execute the command across 4 tasks (one per GPU)
mpirun -np 4 python -m torch.distributed.launch \
    --nproc_per_node=4 main.py \
    --model $model \
    --prune_method "wanda" \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "2:4" \
    --save "out/llama_7b/2-4/wanda/"

echo "Finished wanda pruning with MPI on a single node"
