#!/bin/bash

case="result_20180926_lr0001_ep3"
# running script 
python ../src/main_typhoon_fastai.py --result_path $case --learning_rate 0.001 --n_epochs 3 >& log-$case

case="result_20180926_lr0001_ep10"
# running script 
python ../src/main_typhoon_fastai.py --result_path $case --learning_rate 0.001 --n_epochs 10 >& log-$case

case="result_20180926_lr001_ep3"
# running script 
python ../src/main_typhoon_fastai.py --result_path $case --learning_rate 0.01 --n_epochs 3 >& log-$case

case="result_20180926_lr0001_size128"
# running script 
python ../src/main_typhoon_fastai.py --result_path $case --learning_rate 0.001 --n_epochs 3 --size_img 128 >& log-$case
