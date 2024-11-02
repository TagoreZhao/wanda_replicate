#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=llama2-pruning
#SBATCH --output=logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4  # Allocate 4 CPU cores per task

module load mpi/openmpi-x86_64

# Detect available GPUs
available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s "," 0 $((available_gpus - 1)))

# Set common variables
model="meta-llama/Llama-2-7b-chat-hf"
sparsity_ratio=0.5
export MASTER_PORT=$((12000 + RANDOM % 1000))  # Unique port for each job
export OMP_NUM_THREADS=1

echo "Running wanda pruning with MPI on $available_gpus available GPUs"

# Run pruning with torchrun
mpirun --oversubscribe -np $available_gpus torchrun \
    --nproc_per_node=$available_gpus \
    --rdzv_conf join_timeout=100000 \
    --max_restarts 5 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:${MASTER_PORT} \
    main.py \
    --model $model \
    --prune_method "wanda" \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type "2:4" \
    --save "out/llama_7b/2-4/wanda/"

wait
echo "Finished wanda pruning with MPI on $available_gpus GPUs"
exit 0
