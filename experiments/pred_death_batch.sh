#!/bin/bash

for i in {1..10}
do
  sbatch --job-name=dth$i ~/project/wearables/experiments/pred_death.sh $i
done
