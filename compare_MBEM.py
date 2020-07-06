# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:01:28 2020

@author: Zhe Cao
"""

import numpy as np
import matplotlib.pyplot as plt

vali_accs_ori = np.load('MNIST PWF COPY.npy')
vali_accs_Z = np.load('MNIST PWF COPY MBEM_m.npy')
vali_accs_J = np.load('MNIST PWF COPY MBEM_m-2.npy')

copy_rate_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def plot_acc_mul(vali_accs, copy_rate_range, ver, num_iters=1, title=None, filename=None):
    avg = vali_accs.mean(axis=0)
    min_acc = vali_accs.min(axis=0)
    max_acc = vali_accs.max(axis=0)
    lower = avg - min_acc
    upper= max_acc - avg
    plt.errorbar(x=copy_rate_range, y=avg[:, 0], yerr=[lower[:, 0], upper[:, 0]], label=ver+"Majority vote", fmt='-o')
    for i in range(1, num_iters+1):
        plt.errorbar(x=copy_rate_range, y=avg[:, i], yerr=[lower[:, i], upper[:, i]], label=ver+"Iteration {}".format(i), fmt='-o')
    plt.ylim(min(0.95, vali_accs.min()-0.01), 1.01)
    # plt.title(title)
    plt.xlabel("Copy probability")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    
plot_acc_mul(vali_accs_ori, copy_rate_range, "MBEM ")
plot_acc_mul(vali_accs_Z, copy_rate_range, "MBEM-Zhe ")
plot_acc_mul(vali_accs_J, copy_rate_range, "MBEM-Josh ")

plt.savefig("Compare MBEM", bbox_inches='tight')
plt.show()