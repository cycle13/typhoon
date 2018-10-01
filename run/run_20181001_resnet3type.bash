#!/bin/bash

# python env
source activate fastai

# running script 
case="result_20181001_red_resnext50_sz299"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network resnext50 --batch_size 12 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --size_img 299 >& log-$case

# running script 
case="result_20181001_red_resnet50_sz224"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network resnet50 --batch_size 32 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --size_img 224 >& log-$case

# running script 
case="result_20181001_red_resnet152_sz224"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --network resnet152 --batch_size 32 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --size_img 224 >& log-$case

