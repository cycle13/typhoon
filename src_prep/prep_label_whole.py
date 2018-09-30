#
# Prepare label csv for classification
#
import pandas as pd
import numpy as np
import os
import random

def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    """ Get a list of index values for Validation set from a dataset
    
    Arguments:
        n : int, Total number of elements in the data set.
        cv_idx : int, starting index [idx_start = cv_idx*int(val_pct*n)] 
        val_pct : (int, float), validation set percentage 
        seed : seed value for RandomState
        
    Returns:
        list of indexes 
    """
    np.random.seed(seed)
    n_val = int(val_pct*n)
    idx_start = cv_idx*n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start+n_val]
    
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

    # training / validaton split 20% validation
    id_va_po = sorted(get_cv_idxs(len(files_po)))
    id_va_ng = sorted(get_cv_idxs(len(files_ng)))

    files_po_va = [files_po[x] for x in id_va_po]
    files_po_tr = [files_po[x] for x in list(set(range(len(files_po)))-set(id_va_po))]
    files_ng_va = [files_ng[x] for x in id_va_ng]
    files_ng_tr = [files_ng[x] for x in list(set(range(len(files_ng)))-set(id_va_ng))]

    # resample positve instances to match negative ones
    files_ng_lst = [files_ng_tr,files_ng_va]
    files_po_lst = [files_po_tr,files_po_va]

    df_list = []
    for i in range(2):
        files_ng = files_ng_lst[i]
        files_po = files_po_lst[i]
        Ntr = len(files_ng)
    
        ng_slct = files_ng
        ng_slct2 = ['nonTC/'+x for x in ng_slct]

        # resample
        po_slct = random.choices(files_po,k=Ntr)
        po_slct2 = ['TC/'+x for x in po_slct]

        # save to file
        df_list.append(pd.DataFrame({ 'fname' : po_slct2 + ng_slct2,
                                      'label' : ([1]*Ntr)+([0]*Ntr)}))

    df = pd.concat(df_list)
    df.to_csv('../data/label_list_whole_v2.csv',header=False, index=False)
    # validation label
    valid_label =  ([0]*len(df_list[0])) + ([1]*len(df_list[1]))
    valid_idx = np.where(valid_label)[0]
    df_idx = pd.DataFrame({'valid_idx' :valid_idx})
    df_idx.to_csv('../data/valid_idx_whole_v2.csv',header=False, index=False)

