#!/bin/bash

# python env
source activate fastai

# running script 
case="result_20181005_whl_resnet50_rot"
python ../src/main_typhoon_withcsv_whl.py --result_path $case --train_flist ../data/label_list_whole_v2.csv --network resnet50 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 1 --transform rotate --size_img 128 >& log-$case

case="result_20181005_whl_vgg19"
python ../src/main_typhoon_withcsv_whl.py --result_path $case --train_flist ../data/label_list_whole_v2.csv --network vgg19 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 1 --transform rotate --size_img 128 >& log-$case


case="result_20181005_whl_resnext50"
python ../src/main_typhoon_withcsv_whl.py --result_path $case --train_flist ../data/label_list_whole_v2.csv --network resnext50 --batch_size 16 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 1 --transform rotate --size_img 128 >& log-$case


