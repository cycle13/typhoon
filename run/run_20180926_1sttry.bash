#!/bin/bash

case="result_20180926_1sttry"

# running script 
python ../src/main_typhoon_fastai.py --result_path $case --learning_rate 0.001 --n_epochs 3 >& $case/log-$case 

