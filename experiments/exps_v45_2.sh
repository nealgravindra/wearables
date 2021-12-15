#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wearables
mkdir -p ./jobs/
counter=0
for i in {4..6}
do
  echo "starting "$1"_n"$i" on cuda:"$2""
  python /home/ngrav/project/wearables/scripts/exps_v45.py --exp="$1" --trial=$i --cuda_nb=$2 2>&1 > ./jobs/"$1"_n"$i".log
  echo "done with ""$1"_n"$i"
  let counter++
done
mail -s "nalab4_jobs" ngravindra@gmail.com <<< "done with $1 for $counter trials"
exit
