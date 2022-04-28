import os
import random
from statistics import mean

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from munch import Munch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def concat_config(arg, config):
    config = Munch(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['seed'] = arg.seed
    config['name'] = arg.name
    if not os.path.exists(os.path.join(os.getcwd(), arg.name)):
        dir_name = os.path.join(os.getcwd(), arg.name)
        os.makedirs(dir_name)
    else:
        i = 0
        dir_name = os.path.join(os.getcwd(), arg.name + f'_{i}')
        while os.path.exists(dir_name):
            i += 1
            dir_name = os.path.join(os.getcwd(), arg.name + f'_{i}')
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)
    config['save_dir'] = dir_name
    config['device'] = device
    config['data_dir'] = arg.data_dir
    config['viz_log'] = arg.viz_log
    config['metric'] = arg.metric
    config['loss'] = arg.loss
    config['save_interval'] = arg.save_interval

    return config


def load_config(args):
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)
    config = concat_config(args, config)
    return config


def get_metrics(output, mask):
    with torch.no_grad():
        output_met = torch.argmax(F.softmax(output, dim=1), dim=1) - 1
        mask_met = mask - 1
        tp, fp, fn, tn = smp.metrics.get_stats(output_met, mask_met, mode='multiclass', num_classes=10, ignore_index=-1)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
    return f1_score, recall, precision


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu


def add_hist(hist, label_trues, label_preds, n_class):
    """
        stack hist(confusion matrix)
    """
    with torch.no_grad():
        label_preds = torch.argmax(label_preds, dim=1).detach().cpu().numpy()
        label_trues = label_trues.detach().cpu().numpy()
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist




def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist