#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wearables
mkdir -p ./jobs/

for i in {1..6}
do
  echo "staring "$1"_n"$i" on cuda:"$2""
  python -u /home/ngrav/project/wearables/scripts/exp_v43.py --exp="$1" --trial=$i --cuda_nb=$2 2>&1 > ./jobs/"$1"_n"$i".log
  echo "done with ""$1"_n"$i"
done
mail -s "nalab3_jobs" ngravindra@gmail.com <<< "done with $1 for all trials"
exit
