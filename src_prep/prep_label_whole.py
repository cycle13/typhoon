#
# Prepare label csv for classification
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
    # test data
    te_dir = '../data/test'

    files_po = os.listdir(tr_dir_po)
    files_ng = os.listdir(tr_dir_ng)
    files_te = os.listdir(te_dir)
    
    print('training data TC size',len(files_po))
    print(files_po[0])
    print('training data nonTC size',len(files_ng))
    print(files_ng[0])
    print('test data size',len(files_te))
    print(files_te[0])

    # resample positve instances to match negative ones
    Ntr = len(files_ng)
    
    ng_slct = files_ng
    ng_slct2 = ['nonTC/'+x for x in ng_slct]

    # resample
    po_slct = random.choices(files_po,k=Ntr)
    po_slct2 = ['TC/'+x for x in po_slct]

    # save to file
    df = pd.DataFrame({ 'fname' : po_slct2 + ng_slct2,
                        'label' : ([1]*Ntr)+([0]*Ntr)})
    df.to_csv('../data/label_list_whole.csv',header=False, index=False)
