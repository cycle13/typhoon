#!/bin/bash

# python env
source activate fastai

# running script 
case="result_20180931_red_vgg19"
python ../src/post_pred_typhoon_withcsv.py --model_path result_20180931_red_vgg19 --result_path $case --train_flist ../data/label_list_reduced.csv --test_path test/ --network vgg19 --batch_size 64 --learning_rate 0.01 --lr_diff 1.0 --n_cycles 5 --size_img 128 #>& log-$case

