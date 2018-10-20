#
# Prepare dir for not sampled smaples
#
import pandas as pd
import os
import random
    
if __name__ == '__main__':
    # fixed seed
    random.seed(101)
    
    # training data
    tr_dir_po = '../data/train_orig/TC'
    tr_dir_ng = '../data/train_orig/nonTC'

    files_po = os.listdir(tr_dir_po)
    files_ng = os.listdir(tr_dir_ng)
    
    print('training data TC size',len(files_po))
    print(files_po[0])
    print('training data nonTC size',len(files_ng))
    print(files_ng[0])

    # reduce negative instances to match positve ones
    Ntr = len(files_po)
    
    ng_slct = random.sample(files_ng,Ntr)
    # create a list of "Not-Used" negative instances
    ng_noslct = sorted(set(files_ng) - set(ng_slct))

    print('creating symbolic link')
    for fname in ng_noslct:
        path1 = '/home/tsuyoshi/typhoon/data/train_orig/nonTC/' + fname
        path2 = '/home/tsuyoshi/typhoon/data/unused_noTC/' + fname
        os.symlink(path1,path2)

    # save to file
    #df = pd.DataFrame({ 'fname' : po_slct2 + ng_slct2,
    #                    'label' : ([1]*Ntr)+([0]*Ntr)})
    #df.to_csv('../data/label_list_reduced.csv',header=False, index=False)
