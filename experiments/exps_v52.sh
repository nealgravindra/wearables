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
  for i in {0.05,0.1,0.2}
  do
    echo "starting "$var"_n"$i" on cuda:"$var""
    python /home/ngrav/project/wearables/scripts/exps_v51.py --exp="biasvarp""$i" --trial=1 --cuda_nb=var > ./jobs/"$var"_n"$i".log
    echo "done with p=""$i"
    let counter++
  done
  mail -s "nalab4_jobs" ngravindra@gmail.com <<< "done with $var for $counter trials"
  exit
elif [ $var -eq $gpu1 ]
then
  for i in {0.3,0.4,0.6}
  do
    echo "starting "$var"_n"$i" on cuda:"$var""
    python /home/ngrav/project/wearables/scripts/exps_v51.py --exp="biasvarp""$i" --trial=1 --cuda_nb=$var > ./jobs/"$var"_n"$i".log
    echo "done with p=""$i"
    let counter++
  done
  mail -s "nalab4_jobs" ngravindra@gmail.com <<< "done with $var for $counter trials"
  exit
elif [ $var -eq $gpu2 ]
then
  for i in {0.7,0.8,0.9}
  do
    echo "starting "$var"_n"$i" on cuda:"$var""
    python /home/ngrav/project/wearables/scripts/exps_v51.py --exp="biasvarp""$i" --trial=1 --cuda_nb=$var > ./jobs/"$var"_n"$i".log
    echo "done with p=""$i"
    let counter++
  done
  mail -s "nalab4_jobs" ngravindra@gmail.com <<< "done with $var for $counter trials"
  exit
fi 
