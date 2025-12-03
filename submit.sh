#!/bin/bash
#
#SBATCH --job-name=benchopt-pca
#SBATCH --output=res_benchopt_pca_%A_%a.txt
#
#SBATCH --ntasks=9
#SBATCH --partition=parietal,normal
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --mem=6G
#SBATCH --error error_%A_%a.out
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate benchopt-pca

cd ~/benchmarks/benchmark_pca
python -m benchopt run . --config config.yml
