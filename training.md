
Slurm Training Command
```bash
srun -p gpucluster --job-name=llama2-pruning --output=logs/slurm-%j.out ./llama2-7b.sh
```

Check GPU Status
```bash
srun -p gpucluster nvidia-smi
```