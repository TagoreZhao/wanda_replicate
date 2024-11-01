
Slurm Training Command
```bash
srun -p gpucluster --job-name=llama2-pruning --output=logs/slurm-%j.out ./llama2-7b.sh
```

```bash
sbatch test.sh
```

Show GPU status for every second, so that you can constantly monitor the gpu status
```bash
srun -p gpucluster nvidia-smi -l 1
```