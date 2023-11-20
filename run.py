import importlib
import argparse
from utils.utils import *

MODEL_DIR = None
DATA_DIR = 'data/'
PROJECT = 'mics'
PHASE = 'inc'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about base config
    parser.add_argument('-phase', type=str, default=PHASE)
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100', 'tiered_imagenet'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-gpu', default=0)
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-memo', type=str, default='')


    # about training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-lr_cf', type=float, default=0)
    parser.add_argument('-schedule', type=str, default='Step', choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos', 'euclidean', 'euclidean-squared'],
                        help='euclidean means L2 norm, euclidean-squared means squared L2 norm')
    # ft_dot means using linear classifier, ft_cos means using cosine classifier
    # avg_cos means using average data embedding and cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'euclidean', 'avg_dot', 'avg_cos', 'avg_euclidean', 'euclidean-squared'])
    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')

    # about mics
    parser.add_argument('-st_ratio', type=float, default=1)  # session trainable parameter ratio
    parser.add_argument('-train', type=str, default='vanilla',
                        choices=['vanilla', 'autoaug', 'mixup_hidden', 'autoaugmix', 'cutmix', 'mixup_hidden_similar'])
    parser.add_argument('-mixup_alpha', type=float, default=0.0, help='alpha parameter for mixup')
    parser.add_argument('-is_autoaug', type=str2bool, default=False)
    parser.add_argument('-use_hard_positive_aug', type=str2bool, default=False)
    parser.add_argument('-hpa_type', type=str, default='none',
                        choices=['none', 'inter_class', 'intra_class', 'both'])
    parser.add_argument('-add_noise_level', type=float, default=0.)
    parser.add_argument('-mult_noise_level', type=float, default=0.)
    parser.add_argument('-minimum_lambda', type=float, default=0.5)
    parser.add_argument('-label_sharpening', type=str2bool, default=True)
    parser.add_argument('-label_mix', type=str, default='vanilla',
                        choices=['vanilla', 'steep_other', 'steep_dummy',
                                 'exp_dummy', 'gaussian_dummy', 'sine_dummy', 'piecewise_linear_dummy'])
    parser.add_argument('-label_mix_threshold', type=float, default=0.2)
    parser.add_argument('-gaussian_h1', type=float, default=0.2)
    parser.add_argument('-piecewise_linear_h1', type=float, default=0.5)
    parser.add_argument('-piecewise_linear_h2', type=float, default=0.)
    parser.add_argument('-num_similar_class', type=int, default=3)
    parser.add_argument('-num_pre_allocate', type=int, default=40)
    parser.add_argument('-normalized_middle_classifier', type=str2bool, default=True)
    parser.add_argument('-exp_coef', type=float, default=1.)
    parser.add_argument('-drop_last', type=str2bool, default=False)
    parser.add_argument('-use_resnet_alice', type=str2bool, default=False)
    parser.add_argument('-use_mixup', type=str2bool, default=True)
    parser.add_argument('-use_softlabel', type=str2bool, default=True)
    parser.add_argument('-use_midpoint', type=str2bool, default=True)

    return parser


def main(args):
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.%s_trainer' % (args.project, args.phase)).FSCILTrainer(args)
    trainer.train()


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()

    main(args)
