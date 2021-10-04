#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=00:05:00
#SBATCH --job-name=test_2nodes
#SBATCH --output=%x.out
#SBATCH -p compute_full_node

nvidia-smi
