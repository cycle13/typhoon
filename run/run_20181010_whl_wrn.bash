#!/bin/bash

# python env
source activate fastai

# running script 
case="result_20181010_whl_wrn50_lr001"
python ../src/main_typhoon_withcsv_whl.py --result_path $case --train_flist ../data/label_list_whole_v2.csv --network wrn50 --batch_size 32 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 1 --transform rotate --size_img 128 #>& log-$case

#case="result_20181010_whl_wrn50_lr0002"
#python ../src/main_typhoon_withcsv_whl.py --result_path $case --train_flist ../data/label_list_whole_v2.csv --network wrn50 --batch_size 32 --learning_rate 0.002 --lr_diff 1.0 --n_cycles 1 --transform rotate --size_img 128 >& log-$case

#case="result_20181010_whl_wrn50_lr005"
#python ../src/main_typhoon_withcsv_whl.py --result_path $case --train_flist ../data/label_list_whole_v2.csv --network wrn50 --batch_size 32 --learning_rate 0.05 --lr_diff 1.0 --n_cycles 1 --transform rotate --size_img 128 >& log-$case






