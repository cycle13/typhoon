#!/bin/bash

# python env
source activate fastai

# running script 
case="result_20180930_withcsv_resnet34"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network resnet34 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 4 --size_img 128 >& log-$case

case="result_20180930_withcsv_resnet50"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network resnet50 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 4 --size_img 128 >& log-$case

case="result_20180930_withcsv_resnet152"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network resnet152 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 4 --size_img 128 >& log-$case

case="result_20180930_withcsv_resnext50"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network resnext50 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 4 --size_img 128 >& log-$case

case="result_20180930_withcsv_vgg19"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network vgg19 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 4 --size_img 128 >& log-$case

case="result_20180930_withcsv_inception"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network inceptionresnet_2 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 4 --size_img 128 >& log-$case
