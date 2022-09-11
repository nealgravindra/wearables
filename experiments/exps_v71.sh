#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wearables
mkdir -p ./jobs/
counter=0
declare -i var=$1
declare -i gpu0=0
declare -i gpu1=1
declare -i gpu2=2
if [ $var -eq $gpu0 ]
then 
  for i in {1,2,3}
  do
    echo "starting "$var"_n"$i" on cuda:"$var""
    python /home/ngrav/project/wearables/scripts/exps_v71.py --exp="itv71" --trial=$i --cuda_nb=$var > ./jobs/"itv71"_n"$i".log
    echo "done with n=""$i"
    let counter++
  done
  mail -s "nalab4_jobs" ngravindra@gmail.com <<< "done with $var for $counter trials"
  exit
elif [ $var -eq $gpu1 ]
then
  for i in {4,5,6}
  do
    echo "starting "$var"_n"$i" on cuda:"$var""
    python /home/ngrav/project/wearables/scripts/exps_v71.py --exp="itv71" --trial=$i --cuda_nb=$var > ./jobs/"itv71"_n"$i".log
    echo "done with n=""$i"
    let counter++
  done
  mail -s "nalab4_jobs" ngravindra@gmail.com <<< "done with $var for $counter trials"
  exit
elif [ $var -eq $gpu2 ]
then
  for i in {7,8,9}
  do
    echo "starting "$var"_n"$i" on cuda:"$var""
    python /home/ngrav/project/wearables/scripts/exps_v71.py --exp="itv71" --trial=$i --cuda_nb=$var > ./jobs/"itv71"_n"$i".log
    echo "done with n=""$i"
    let counter++
  done
  mail -s "nalab4_jobs" ngravindra@gmail.com <<< "done with $var for $counter trials"
  exit
fi 
