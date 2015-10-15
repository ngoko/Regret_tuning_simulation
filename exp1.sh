#!/bin/bash
  #SBATCH --job-name= autotuning/activelearning
  #SBATCH --output=slurm.out
  #SBATCH --error=slurm.err
  #SBATCH --nodes=1
  #SBATCH --ntasks-per-node=1

srun -l python exp1.py
