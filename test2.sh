#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=llama2-pruning
#SBATCH --output=logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4    # Allocate 4 CPU cores per GPU task

module load mpi/openmpi-x86_64

# Detect the available GPUs
available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

# Adjust process count to match available GPUs
if [ "$available_gpus" -gt 4 ]; then
  num_processes=4
elif [ "$available_gpus" -gt 0 ]; then
  num_processes=$available_gpus
else
  echo "No GPUs available."
  exit 1
fi

# Common variables
model="meta-llama/Llama-2-7b-chat-hf"
sparsity_ratio=0.5
export MASTER_PORT=$((12000 + RANDOM % 1000))  # Unique port for each job
export OMP_NUM_THREADS=1

# Run pruning with torchrun
echo "Running wanda pruning with MPI on a single node with $num_processes GPUs"

mpirun -np $num_processes torchrun \
    --nproc_per_node $num_processes \
    --rdzv_conf join_timeout=50000 \
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

echo "Finished wanda pruning with MPI on a single node"
# Explicitly exit
exit 0
