# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:58:01 2020

@author: Zhe Cao

Balance vs imbalance
Varied skill levels
Updated on 27 Aug
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader
from workers import *
from utils import *
from train_mbem import call_train as call_train_mbem
from MBEM import posterior_distribution
from train import call_train
import models
import gc

#%% Functions
def plot_result_std_cr_gammax(arrays, gamma_range, title, ylabel, filename):
    """
    arrays: num_rep * num_copyrates * 2
    standard deviation
    skill levels as x axis
    """
    avg = arrays.mean(axis=0)
    std = arrays.std(axis=0)
    lower = std
    upper= std
    plt.errorbar(x=gamma_range, y=avg[:, 0], yerr=[lower[:, 0], upper[:, 0]], capsize=4, label="Unbalanced", fmt='--o')
    plt.errorbar(x=gamma_range, y=avg[:, 1], yerr=[lower[:, 1], upper[:, 1]], capsize=4, label="Balanced", fmt='--o')
    plt.ylim(0.0, arrays.max()+0.01)
    plt.title(title)
    plt.xlabel("Skill level")
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(filename+".png", bbox_inches='tight')
    np.save(filename+".npy", arrays)
    plt.show()

#%% Main
if __name__ == "__main__":
    dataset = 'mnist'
    #%% Load cifar-10 data
    if dataset == 'cifar10':    
        X = np.load("./cifar10/X.npy")
        y = np.load("./cifar10/y.npy")
        X_test = np.load("./cifar10/X_test.npy")
        y_test = np.load("./cifar10/y_test.npy")        
        k = 10
        X = X / 255.0
        X_test = X_test / 255.0
        y = y.astype(int)
        y_test = y_test.astype(int)
        y = np.eye(k)[y]
        y_test = np.eye(k)[y_test]
        X_train, X_vali = X[:45000], X[45000:]
        y_train, y_vali = y[:45000], y[45000:]
        use_aug = False    
    #%% Load MNIST data
    elif dataset == 'mnist':
        X = np.load("./mnist/X.npy")
        y = np.load("./mnist/y.npy", allow_pickle=True)
        k = 10
        X = X / 255.0
        X = X.reshape(-1, 1, 28, 28)
        y = y.astype(int)
        y = np.eye(k)[y]
        X_train, X_vali, X_test = X[:50000], X[50000:60000], X[60000:]
        y_train, y_vali, y_test = y[:50000], y[50000:60000], y[60000:]
        use_aug = False
    else:
        raise ValueError("Invalid value for dataset!")

    m = 5 # number of users
    gamma_range = np.arange(0.30, 0.501, 0.05)
    repeat = 2 # redundancy
    if dataset == 'cifar10':
        valid_range = np.arange(45000)
    elif dataset == 'mnist':
        valid_range = np.arange(50000) # max. 50000
    else:
        pass
    print("Training on {} samples.".format(len(valid_range)))
    num_busy = 1 # 1 by default, starting from No.0
    copy_rates = np.zeros(m)
    copy_ids = np.arange(1, 2) # user no.1 is the copycat
    copy_rate = 0.0
    copy_rates[copy_ids] = copy_rate
    num_rep = 3 # repetition
    test_accs = np.zeros((num_rep, len(gamma_range), 2)) # four algorithms to compare
    conf_errors = np.zeros((num_rep, len(gamma_range), 2))
    cp_errors = np.zeros((num_rep, len(gamma_range), 2))
    
    title = "Redundancy:{}, copy probability: {}".format(repeat, copy_rate)
    filename = "Imbalance_r_{}_c_{}_gammax".format(repeat, copy_rate).replace('.', '')
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    filename = '/'.join(['.', result_dir, filename])
    
    for rep in range(num_rep):
        print("Repetition: {}".format(rep))
        for i, gamma in enumerate(gamma_range):
            print("----------------")
            print("Skill level: {}".format(gamma))
            conf = generate_conf_pairflipper(m, k, gamma)
            
            # 1. Imbalance setting + vanilla model
            labels_train, _, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
            labels_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=False, model=None, use_aug=use_aug, est_cr=False, reweight=False, dataset=dataset)
            test_accs[rep, i, 0] = test_acc
            conf_errors[rep, i, 0] = conf_error
            # cp_errors[rep, i, 0] = cp_error # not applicable
            plot_conf_mat_h(est_conf, conf)
            
            # 2. Balanced setting + vanilla model
            labels_train, _, workers_on_example = generate_labels_weight(y_train[valid_range], repeat, conf)
            labels_vali, _, workers_on_example_vali = generate_labels_weight(y_vali, repeat, conf)
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=False, model=None, use_aug=use_aug, est_cr=False, reweight=False, dataset=dataset)
            test_accs[rep, i, 1] = test_acc
            conf_errors[rep, i, 1] = conf_error
            # cp_errors[rep, i, 1] = cp_error # not applicable
            plot_conf_mat_h(est_conf, conf)
      
            
    plot_result_std_cr_gammax(test_accs, gamma_range, title=title, ylabel="Test accuracy", filename=filename+"_testacc")
#     plot_result_std_cr_gammax(cp_errors, gamma_range, title=title, ylabel="Copy probability estimation error", filename=filename+"_cperror")
    plot_result_std_cr_gammax(conf_errors, gamma_range, title=title, ylabel="Confusion matrix estimation error", filename=filename+"_conferror")
