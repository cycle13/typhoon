#!/bin/bash

case="result_20180928_lr001_ep4"
# running script 
python ../src/main_typhoon_fastai.py --result_path $case --learning_rate 0.01 --n_epochs 4 --size_img 128 >& log-$case

