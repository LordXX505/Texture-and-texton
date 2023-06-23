import os
import datetime
import shutil
import torch
import torch.nn as nn
import torchvision.utils as vutils
import random
import numpy as np
import logging
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_output_dir(exp_id, test):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/{}'.format(exp_id), t)
    if test == 0:
        return output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir+'/samples')
        os.makedirs(output_dir + '/ckpt')
    return output_dir

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def save_images(img, path,nrow):
    vutils.save_image(img, path, normalize=True, nrow=nrow)

def set_gpu(gpuid):
    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:{}'.format(gpuid))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed

def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def to_named_dict(ns):
    d = AttrDict()
    for (k, v) in zip(ns.__dict__.keys(), ns.__dict__.values()):
        d[k] = v
    return d



def plot_stats(output_dir, stats, interval):
    content = stats.keys()
    f, axs = plt.subplots(len(content), 1, figsize=(20, len(content) * 5))
    for j, (k, v) in enumerate(stats.items()):
        if len(content) ==1:
            axs.plot(interval, v)
            axs.set_ylabel(k)
        else:
            axs[j].plot(interval, v)
            axs[j].set_ylabel(k)
    f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, 'stat.png'), bbox_inches='tight')
    plt.close(f)