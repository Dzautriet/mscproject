# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:51:16 2020

@author: Zhe Cao
"""

import numpy as np
import matplotlib.pyplot as plt

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
    mse = ((est_conf - conf)**2).mean()
    print("Mean squared error: {:.5f}".format(mse))
        