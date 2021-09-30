#!/bin/bash

for ((i=1; i<=1; i++))
do
  for j in {0..376}
    do
      sbatch --job-name=IT$i_$j --output=./logs/ITv3_allmd_"$i"_"$j".log /home/ngr4/project/wearables/experiments/ITv3_allmd.slurm $i $j
    done
done
