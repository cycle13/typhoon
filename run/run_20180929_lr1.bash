#!/bin/bash

# python env
source activate fastai

# running script 
case="result_20180929_withcsv_lrdecay1"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 3 --size_img 128 >& log-$case
