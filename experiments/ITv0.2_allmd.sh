#!/bin/bash

for i in {1}
do
  for j in {0..103}
    do
      sbatch --job-name=IT$i_$j ~/project/wearables/experiments/ITv0.2_allmd.py.sh "InceptionTimev0.2_allmd" $i $j
    done
done
