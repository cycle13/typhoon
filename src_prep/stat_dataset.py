# 
# take statistics for the whole dataset
# 

import os
import sys
import time

import numpy as np
import pandas as pd

# -----------------------------
# add "src" as import path
path = os.path.join('../src')
sys.path.append(path)

from fastai.imports import *
from fastai.dataset import *

if __name__ == "__main__":

    tstart = time.time()
    
    PATH = "../data/train_orig/"
    #train_flist = '../data/label_list_whole_v2.csv'
    sz = 16

    # training data
    tr_dir_po = '../data/train_orig/TC'
    tr_dir_ng = '../data/train_orig/nonTC'
    files_po = os.listdir(tr_dir_po)
    files_ng = os.listdir(tr_dir_ng)
    files_po = ['TC/'+x for x in files_po]
    files_ng = ['nonTC/'+x for x in files_ng]

    # save to file
    df_train = pd.DataFrame({ 'fname' : files_po + files_ng,
                              'label' : ([1]*len(files_po))+([0]*len(files_ng))})
    #df_train = pd.read_csv(train_flist,header=None)
    
    #N = 100             
    #N = 40000          
    N = df_train.shape[0] #2244223
    
    df_train = df_train.sample(n=N,random_state=0,replace=False)
    df_train = df_train.reset_index(drop=True)

    Nseps = 100
    X = np.zeros(Nseps,dtype=int)
   
    for n in range(N):
        fn = df_train.loc[n,'fname']
        label = df_train.loc[n,'label']
        if n % 1000 == 0:
            print(n,fn,label)
        # read tiff file
        img = Image.open(PATH+str(fn))
        im = np.asarray(img)
        his = np.histogram(im.flatten(),bins=Nseps,range=(0,2))
        # add to 
        X = X + his[0]

    grd = 0.5*(his[1][:-1] + his[1][1:])
    df_cnt = pd.DataFrame({ 'x' : np.round(grd,2),
                            'count' : X})
    df_cnt.to_csv('../data/hist_all_data.csv',index=False)

    tend = time.time()
    tdiff = float(tend-tstart)
    print('Elapsed time[s]: %f \n' % tdiff)






