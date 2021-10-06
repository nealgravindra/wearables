#!/bin/bash

mkdir -p ./logs
for i in {1..10}
do
    sbatch --job-name=ga_$i --output=./logs/ITv3_GA_n"$i".log /home/ngr4/project/wearables/experiments/ITv3_GA.slurm "ITv3_GA" $i
done
