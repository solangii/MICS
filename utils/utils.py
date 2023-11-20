import random
import torch
import os
import time
import csv
from torch.autograd import Variable
import numpy as np
import pprint as pprint

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def count_mix_acc(logits, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def print_table(results, args):
    print("{:<13}{}".format("Pretraining:", args.base_mode.split('_')[-1]))
    if args.lr_new != 0:
        print("{:<13}{}".format("Fine-tuning:", args.new_mode.split('_')[-1]))
    print("{:<13}{}".format("Similarity:", args.new_mode.split('_')[-1]))

    loss = {}
    if args.use_ce: loss['ce'] = args.lambda_ce
    if args.use_tl: loss['triple'] = args.lambda_tl
    if args.use_cl: loss['cossim'] = args.lambda_cl
    if args.use_rl: loss['l1'] = args.lambda_rl
    if args.use_pl:
        if args.pl_opt == 'basic': loss['push(basic))'] = args.lambda_pl
        elif args.pl_opt == 'simple': loss['push(simple)'] = args.lambda_pl
    if args.use_sl:
        if args.sl_opt == 'basic': loss['spacing(basic)'] = args.lambda_sl
        elif args.sl_opt == 'simple': loss['spacing(simple)'] = args.lambda_sl

    print("{:<13}{}".format("Loss:", ", ".join(str(k) for k in loss.keys())))
    print("{:<13}{}".format("Lambda:", ", ".join(str(k) for k in loss.values())))

    str_head = "{:<9}".format('')
    str_acc = "{:<9}".format('Acc:')
    str_acc_base = "{:<9}".format('Base:')
    str_acc_novel = "{:<9}".format('Novel:')
    str_acc_old = "{:<9}".format('Old:')
    str_acc_new = "{:<9}".format('New:')

    for i in range(len(results['acc'])):
        str_head = str_head + "{:<9}".format('sess' + str(int(i + 1)))
        str_acc = str_acc + "{:<9}".format(str(round(results['acc'][i] * 100.0, 1)) + "%")
        str_acc_base = str_acc_base + "{:<9}".format(str(round(results['acc_base'][i] * 100.0, 1)) + "%")
        str_acc_novel = str_acc_novel + "{:<9}".format(str(round(results['acc_novel'][i] * 100.0, 1)) + "%")
        str_acc_old = str_acc_old + "{:<9}".format(str(round(results['acc_old'][i] * 100.0, 1)) + "%")
        str_acc_new = str_acc_new + "{:<9}".format(str(round(results['acc_new'][i] * 100.0, 1)) + "%")

    print(str_head)
    print(str_acc)
    print(str_acc_base)
    print(str_acc_novel)
    print(str_acc_old)
    print(str_acc_new)
    print('\n')

    str_acc_base2 = "{:<9}".format('Base2:')
    str_acc_novel2 = "{:<9}".format('Novel2:')
    str_acc_old2 = "{:<9}".format('Old2:')
    str_acc_new2 = "{:<9}".format('New2:')

    for i in range(len(results['acc'])):
        str_acc_base2 = str_acc_base2 + "{:<9}".format(str(round(results['acc_base2'][i] * 100.0, 1)) + "%")
        str_acc_novel2 = str_acc_novel2 + "{:<9}".format(str(round(results['acc_novel2'][i] * 100.0, 1)) + "%")
        str_acc_old2 = str_acc_old2 + "{:<9}".format(str(round(results['acc_old2'][i] * 100.0, 1)) + "%")
        str_acc_new2 = str_acc_new2 + "{:<9}".format(str(round(results['acc_new2'][i] * 100.0, 1)) + "%")

    print(str_acc_base2)
    print(str_acc_novel2)
    print(str_acc_old2)
    print(str_acc_new2)
    print('\n')