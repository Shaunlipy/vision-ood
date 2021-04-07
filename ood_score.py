import argparse
import numpy as np
import os
import glob
import sys
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
# from metrics import *


def load_dset(npy_path):
    data = np.load(npy_path)
    return data


def get_stat(ind_np, ood_np):
    return np.round(100 * roc_auc_score([1] * len(ind_np) +
                                        [0] * len(ood_np),
                                        np.concatenate([ind_np, ood_np])), 2)



if __name__ == '__main__':
    model = 'deit_tiny_patch16_224'
    ind_data = 'dermnet_train_0_0'
    root_path = glob.glob(f'output/*_{model}_{ind_data}/ood')[0]
    ind = 'dermnet_val_0'
    ood = 'dermnet_unseen_0'

    ind_path = os.path.join(root_path, f'{ind}.npy')
    ood_path = os.path.join(root_path, f'{ood}.npy')
    ind_np = load_dset(ind_path)
    ood_np = load_dset(ood_path)
    val = get_stat(ind_np, ood_np)
    print(val)

