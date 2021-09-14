#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=<ngravindra@gmail.com>

module restore cuda111
conda activate wearables

mkdir -p logs
python -u /home/ngr4/project/wearables/scripts/experiments.py --exp=IT_death --trial="$1"
