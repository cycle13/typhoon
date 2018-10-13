# 
# Undersample "negative" samples using imbalanced learn nearmiss
# 

import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks

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
    
    #N = 1000               #  100000
    #N = 40000               #  100000
    N = df_train.shape[0] #2244223
    
    df_train = df_train.sample(n=N,random_state=0,replace=False)
    df_train = df_train.reset_index(drop=True)
    
    X = np.zeros((N,sz**2))
    print ('X size',sys.getsizeof(X)/1000/1000/1000,'GB')
    y = np.zeros(N,dtype=int)
   
    for n in range(N):
        fn = df_train.loc[n,'fname']
        label = df_train.loc[n,'label']
        if n % 1000 == 0:
            print(n,fn,label)
        # read tiff file
        img = Image.open(PATH+str(fn))
        img = img.resize((sz,sz))
        im = np.asarray(img)
        # interpret greyscale as color image
        # im3 = np.stack([im,im,im],axis=2)
        X[n,:] = im.flatten()
        y[n] = label
   
    # nearmiss 1
    nm = NearMiss(version=1,return_indices=True)
    print("nearmiss calc finished")
    #nm = NearMiss(version=2,return_indices=True)
    #nm = TomekLinks()

    X_resampled, y_resampled, idx_res = nm.fit_sample(X,y)
    df_out = df_train.loc[idx_res]
    df_out.to_csv('../data/label_list_nearmiss1.csv',header=False, index=False)

    tend = time.time()
    tdiff = float(tend-tstart)
    print('Elapsed time[s]: %f \n' % tdiff)






