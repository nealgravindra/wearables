#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wearables
python -u /home/ngrav/project/wearables/scripts/experiments_v42.py --exp="$1" --trial="$2" --cuda_nb=$3 > ./jobs/"$1"_n"$2".log

exit
