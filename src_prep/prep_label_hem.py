#
# Prepare label csv for classification
# Hard example mining
#
import pandas as pd
import os
import random
    
if __name__ == '__main__':
    # fixed seed
    random.seed(101)
    
    # training data
    flabel = '../data/label_list_reduced.csv'
    df_red = pd.read_csv(flabel,header=None)
    df_red.columns = ['fname','label']

    df_red_pos = df_red.loc[df_red['label']==1]
    Npos = len(df_red_pos)
    df_red_neg = df_red.loc[df_red['label']==0]
    Nneg = len(df_red_neg)
    print('Npos,Nneg=',Npos,Nneg)
    Nneg2 = Nneg * 2

    neg_set = set(df_red_neg['fname'].values)

    # read predicted nonTC examples
    f_nonTC = '../run/result_20180931_red_vgg19/post_rerun_nonTC_result_20180931_red_vgg19_prob.tsv'
    df_non = pd.read_table(f_nonTC,sep='\t',header=None)
    df_non.columns = ['fname','prob']
    df_non['prob'] = pd.to_numeric(df_non['prob'], errors='coerce')
    df_non = df_non.sort_values('prob',ascending=False)
    #id_null = pd.to_numeric(df_non['prob'], errors='coerce').isnull()
    #df_non['prob']
      
    for index,row in df_non.iterrows():
        fname = row['fname'].replace('unused_noTC','nonTC')
        neg_set.add(fname)
        #print(len(neg_set))
        if len(neg_set) >= Nneg2 :
            break

    # save to file
    #import pdb;pdb.set_trace()
    lst1 = df_red_pos['fname'].values.tolist() + sorted(neg_set)
    lst2 =  ([1]*Npos) + ([0]*Nneg2)
    df = pd.DataFrame({ 'fname' :lst1, 
                        'label' :lst2})
    df.to_csv('../data/label_list_redhem.csv',header=False, index=False)
