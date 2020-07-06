# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:51:16 2020

@author: Zhe Cao
"""

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

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
        ax_0.set_title("Estimated CM \nfor worker "+str(row))
        ax_0.axis('off')
        ax_1.imshow(conf[row], cmap='Blues', vmin=0, vmax=1)
        ax_1.set_title("Ground Truth CM \nfor worker "+str(row))
        ax_1.axis('off')
    plt.show()
    # mse = ((est_conf - conf)**2).mean()
    # print("Mean squared error: {:.5f}".format(mse))
    mean_diff = np.mean([np.linalg.norm(est_conf[i] - conf[i]) for i in range(m)])
    print("Mean CM est error: {:.4f}.".format(mean_diff))
    
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
        
def accuracy(pred, targets):
    """
    pred: log softmax output
    target: labels
    """
    num_correct = pred.argmax(dim=1).eq(targets.argmax(dim=1)).sum()
    acc = num_correct.float() / targets.size(0)
    return acc
        
        