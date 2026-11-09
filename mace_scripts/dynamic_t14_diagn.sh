#!/bin/bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bcpt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64g
#SBATCH --time=00:05:00
#SBATCH --job-name=dynamis-t214-mace
#SBATCH --output=/u/jsaravananindira/dynamis_torch_214_mace_%j.out

module spider cuda
module spider cudatoolkit
module avail craype-network
module list
command -v nvcc
nvcc --version
nvidia-smi