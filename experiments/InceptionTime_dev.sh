#!/bin/bash

for i in {1..6}
do
  sbatch --job-name=ITR_$i ~/project/wearables/experiments/experiments.py.sh "InceptionTime_dev" $i
done
