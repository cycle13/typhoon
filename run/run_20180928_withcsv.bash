#!/bin/bash

# running script 
case="result_20180928_withcsv_reduced"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_reduced.csv --batch_size 64 --learning_rate 0.01 --n_cycles 3 --size_img 128 >& log-$case

# running script 
case="result_20180928_withcsv_whole"
python ../src/main_typhoon_withcsv.py --result_path $case --train_flist ../data/label_list_whole.csv --batch_size 64 --learning_rate 0.01 --n_cycles 3 --size_img 128 >& log-$case


