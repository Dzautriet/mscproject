# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:51:16 2020

@author: Zhe Cao
"""

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset, DataLoader

class MyDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transforms=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transforms:
            x = self.transforms(x)
        if len(self.tensors) == 2:
            y = self.tensors[1][index]
            return x, y
        else:
            return x

    def __len__(self):
        return self.tensors[0].size(0)
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def correct_rate(y, y_corrupt):
    num_match = y[np.argmax(y, axis=1) == np.argmax(y_corrupt, axis=1)].shape[0]
    return num_match / y.shape[0]

def plot_conf_mat(est_conf, conf):
    """
    Plot estimated and ground truth confusion matrices
    """
    m = conf.shape[0]
    fig, axes = plt.subplots(nrows=m, ncols=2, figsize=(4, 3*m))
    for row, axe in enumerate(axes):
        ax_0, ax_1 = axe
        ax_0.imshow(est_conf[row], cmap='Blues', vmin=0, vmax=1)
        ax_0.set_title("Estimated CM \nfor labeller "+str(row))
        ax_0.axis('off')
        ax_1.imshow(conf[row], cmap='Blues', vmin=0, vmax=1)
        ax_1.set_title("Ground truth CM \nfor labeller "+str(row))
        ax_1.axis('off')
    plt.show()
    # mse = ((est_conf - conf)**2).mean()
    # print("Mean squared error: {:.5f}".format(mse))
    mean_diff = np.mean([np.linalg.norm(est_conf[i] - conf[i]) for i in range(m)])
    print("Mean CM est error: {:.4f}.".format(mean_diff))
    return mean_diff

def plot_conf_mat_h(est_conf, conf):
    """
    Plot estimated and ground truth confusion matrices
    Horizontally
    """
    m = conf.shape[0]
    fig, axes = plt.subplots(nrows=2, ncols=m, figsize=(2.1*m, 2*2))
    fig.tight_layout(pad=1.)
    axes = np.swapaxes(axes, 1, 0)
    fontsize = 14
    for row, axe in enumerate(axes):
        ax_0, ax_1 = axe
        im = ax_0.imshow(conf[row], cmap='Blues', vmin=0, vmax=1)
        ax_0.set_title("Ground truth CM \nof labeller "+str(row), fontsize=fontsize)
        ax_0.axis('off')
        ax_1.imshow(est_conf[row], cmap='Blues', vmin=0, vmax=1)
        ax_1.set_title("Estimated CM \nof labeller "+str(row), fontsize=fontsize)
        ax_1.axis('off')
    # fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.0, 0.1, 0.01, 0.8])
    cb = fig.colorbar(im, cax=cbar_ax, )
    cb.ax.tick_params(labelsize=fontsize)
    # fig.colorbar(im)
    plt.show()
    # mse = ((est_conf - conf)**2).mean()
    # print("Mean squared error: {:.5f}".format(mse))
    mean_diff = np.mean([np.linalg.norm(est_conf[i] - conf[i]) for i in range(m)])
    print("Mean CM est error: {:.4f}.".format(mean_diff))
    return mean_diff
    
def plot_cp(est_copyrates, copy_rates):
    """
    Plot estimated and ground truth confusion matrices
    """
    fig, axe = plt.subplots(nrows=1, ncols=2, figsize=(4, 2))
    ax_0, ax_1 = axe
    ax_0.imshow(est_copyrates[np.newaxis, 1:], cmap='Blues', vmin=0, vmax=1)
    ax_0.set_title("Estimated\n copy rates")
    ax_0.axis('off')
    ax_1.imshow(copy_rates[np.newaxis, 1:], cmap='Blues', vmin=0, vmax=1)
    ax_1.set_title("Ground truth\n copy rates")
    ax_1.axis('off')
    plt.show()    
    
def accuracy(pred, targets):
    """
    pred: log softmax output
    target: labels
    """
    num_correct = pred.argmax(dim=1).eq(targets.argmax(dim=1)).sum()
    acc = num_correct.float() / targets.size(0)
    return acc
        
        