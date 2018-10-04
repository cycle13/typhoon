import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='directory path of data')
    parser.add_argument(
        '--train_flist',
        type=str,
        help='training filelist(csv) path')
    parser.add_argument(
        '--valid_path',
        type=str,
        help='validation filelist(csv) path')
    parser.add_argument(
        '--test_path',
        type=str,
        help='validation filelist(csv) path')
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--network',
        default='resnet34',
        type=str,
        help='Network architecture (supported by fastai)')
    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help='Learning rate')
    parser.add_argument(
        '--lr_diff',
        default=3.0,
        type=float,
        help='Factor for differential learning rate')
    parser.add_argument(
        '--lr_decay',
        default=1.0,
        type=float,
        help='Learning rate decay')
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_cycles',
        default=3,
        type=int,
        help='Number of total cycles to run')
    parser.add_argument(
        '--transform',
        default='top_down',
        type=str,
        help='transformation for data augmentation')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test',
        action='store_true',
        help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--size_img',
        default=64,
        type=int,
        help='Image size to be put into CNN')
    
    args = parser.parse_args()

    return args
