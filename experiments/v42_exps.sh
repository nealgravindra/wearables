#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wearables
for i in {1..6}
do
  python -u /home/ngrav/project/wearables/scripts/experiments_v42.py --exp="$1" --trial=$i --cuda_nb=$2 > ./jobs/"$1"_n"$i".log
  echo "done with ""$1"_n"$i"
done
mail -s "nalab3_jobs" ngravindra@gmail.com <<< "done with $1 for all trials"
exit
