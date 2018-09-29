#
# Typhoon Classification Task (Signate 2018)
# Using 'fastai' library and take csv as input
#
import time
import os
import sys

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

#from sklearn.metrics import recall_score, precision_score

def recall(preds,targs):
    th = 0.5
    prob_TC = np.exp(preds[:,0])
    flg = 1 * (prob_TC > th)
    t2 = targs.cpu().numpy() # ground truth
    f2 = flg.numpy()         # predicted
    TP = sum(t2*f2)
    FN = sum(t2*(1-f2))
    return TP/(TP+FN)

def precision(preds,targs):
    th = 0.5
    prob_TC = np.exp(preds[:,0])
    flg = 1 * (prob_TC > th)
    t2 = targs.cpu().numpy() # ground truth
    f2 = flg.numpy()         # predicted
    TP = sum(t2*f2)
    FP = sum((1-t2)*f2)
    return TP/(TP+FP)

from opts import *

if __name__ == '__main__':
    # parse command-line options
    opt = parse_opts()
    print(opt)

    # create result dir
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # generic log file
    logfile = open(os.path.join(opt.result_path, 'log_run.txt'),'w')
    logfile.write('Start time:'+time.ctime()+'\n')
    tstart = time.time()

    # image size
    sz = opt.size_img

    PATH = "../data/"

    # Neural Net Architecture
    if opt.network == 'resnet34':
        f_model = resnet34
    elif opt.network == 'resnet50':
        f_model = resnet50
    elif opt.network == 'resnet152':
        f_model = resnet152
    elif opt.network == 'resnext50':
        f_model = resnext50
    elif opt.network == 'vgg19':
        f_model = vgg19
    elif opt.network == 'inceptionresnet_2':
        f_model = inceptionresnet_2

    # Transformations
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    
    # Data loader
    print('setting data loader')
    
    n = len(list(open(opt.train_flist)))-1
    print('training data number:',n)
    val_idxs = get_cv_idxs(n)
    
    data = ImageClassifierData.from_csv(path=PATH, folder='train_orig/',
                                        csv_fname=opt.train_flist, bs=opt.batch_size,
                                        tfms=tfms, val_idxs=val_idxs,
                                        test_name='test/', num_workers=6)
    #
    #metrics = [accuracy,recall,precision]
    metrics = [accuracy]
    learn = ConvLearner.pretrained(f_model, data, precompute=False, metrics=metrics)
    #learn = ConvLearner.pretrained(f_model, data, precompute=False)

    # learning rate finder
    flg_finder = False
    if flg_finder == True:
        print('learning rate finder')
        learn.lr_find()
        # save result of lr finder
        df_lr = pd.DataFrame({ 'lr' : learn.sched.lrs,
                               'loss' : learn.sched.losses})
        df_lr.to_csv('%s/lr_finder.csv' % (opt.result_path))

    # from lr finder
    #lr = 0.001
    lr = opt.learning_rate
    # differential learning rate
    lfac = opt.lr_diff
    lrs = np.array([lr/(lfac**2),lr/lfac,lr])

    print('start fitting')
    learn.unfreeze()
    learn.fit(lrs, opt.n_cycles, cycle_len=1, cycle_mult=2)

    # save the trained model
    learn.save(opt.result_path)

    # test with TTA
    print('test with TTA')
    multi_preds, y = learn.TTA(is_test=True)
    preds = np.mean(multi_preds, 0)
    prob_TC = np.exp(preds[:,0])

    test_fnames = learn.data.test_ds.fnames
    # remove 'test/'
    test_fnames2 = [str.replace('test/','') for str in test_fnames]

    for th in [0.4,0.5,0.6,0.7,0.8,0.9,0.97,0.99]:
        flg = 1 * (prob_TC > th)
        df = pd.DataFrame({ 'fname' : test_fnames2,
                            'pred' : flg})
        df = df.sort_values('fname')
        df.to_csv('%s/test_pred_fastai_th%4.2f.tsv' % (opt.result_path,th), header=False, index=False, sep='\t')
        
    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)

