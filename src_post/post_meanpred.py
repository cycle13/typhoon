#
# Prediction by mean/median of predicted values
#

import os
import sys
import pandas as pd
import numpy as np

if __name__ == '__main__':
    cases = [#'result_20180931_red_vgg19',
             #'result_20181001_red_resnet50_sz224',
             #'result_20181003_whole_resnet50_cy1_lr001_ld1',
             #'result_20181005_whl_resnet50_rot',
             #'result_20181007_whl_vgg19',
             #'result_20181010_whl_wrn50_lr001',
             #'result_20181014_whl_dn201_lr001',
             #'result_20181018_whl_resnext101_64',
             'result_20181021_hem_resnet50'
    ]

    synth_dir = '20181021single_hem_resnet50'
    result_path = '../run/result_synthesized/'

    # create result dir
    if not os.path.exists(result_path+synth_dir):
        os.mkdir(result_path+synth_dir)

    M = np.zeros((len(cases),299135))
    
    for i in range(len(cases)):
        case = cases[i]
        print('case:',case)
        df = pd.read_table('../run/'+case+'/pred_'+case+'_prob.tsv',sep='\t',header=None)
        df.columns = ['fname','pred']
        M[i,:] = df.pred

    # calc average prediction
    prob_TC = np.mean(M,axis=0)
    
    # prit out prediction
    for th in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.82,0.83,0.84,0.845,0.85,0.86,0.87,0.88,0.89,0.90]:
        flg = 1 * (prob_TC > th)
        df = pd.DataFrame({ 'fname' : df.fname,
                            'pred' : flg})
        df.to_csv('%s/%s/pred_%s_th%4.3f.tsv' % (result_path,synth_dir,synth_dir,th), header=False, index=False, sep='\t')

    #import pdb;pdb.set_trace()

