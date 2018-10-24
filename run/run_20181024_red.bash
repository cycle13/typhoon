#!/bin/bash

# python env
source activate fastai

# running script 
case="result_20181024_red_resnet50_rot"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network resnet50 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --transform rotate --size_img 128 #>& log-$case

#case="result_20181024_red_wrn50"
#python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network wrn50 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --transform rotate --size_img 128 >& log-$case
#
#case="result_20181024_red_dn201"
#python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network densenet201 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --transform rotate --size_img 128 >& log-$case
#
#case="result_20181024_red_resnext101_64"
#python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network resnext101_64 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --transform rotate --size_img 128 >& log-$case



