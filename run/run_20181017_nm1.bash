#!/bin/bash

# python env
source activate fastai

# running script 
case="result_20181017_nm1_resnet50"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_nearmiss1ps.csv --network resnet50 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --size_img 128 >& log-$case

# running script 
case="result_20181017_nm1_vgg19"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_nearmiss1ps.csv --network vgg19 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --size_img 128 >& log-$case
