#!/bin/bash

# python env
source activate fastai

# running script 
case="result_20181018_whl_resnext101"
python ../src/main_typhoon_withcsv_whl.py --result_path $case --train_flist ../data/label_list_whole_v2.csv --network resnext101 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 1 --transform rotate --size_img 128 >& log-$case

case="result_20181018_whl_resnext101_64"
python ../src/main_typhoon_withcsv_whl.py --result_path $case --train_flist ../data/label_list_whole_v2.csv --network resnext101_64 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 1 --transform rotate --size_img 128 >& log-$case

case="result_20181018_whl_inception_4"
python ../src/main_typhoon_withcsv_whl.py --result_path $case --train_flist ../data/label_list_whole_v2.csv --network inception_4 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 1 --transform rotate --size_img 128 >& log-$case


