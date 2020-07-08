# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:58:01 2020

@author: Zhe Cao

Copycat scenario: systematic comparison
Reweighting mechanism
Ablation study: varied skill levels
"""

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
from train import call_train as call_train_mbem
from MBEM import posterior_distribution
from reg_main_3 import call_train
import models
import gc

#%% Load cifar-10 data
# X = np.load("./cifar10/X.npy")
# y = np.load("./cifar10/y.npy")
# X_test = np.load("./cifar10/X_test.npy")
# y_test = np.load("./cifar10/y_test.npy")

# k = 10
# X = X / 255.0
# X_test = X_test / 255.0
# y = y.astype(int)
# y_test = y_test.astype(int)
# y = np.eye(k)[y]
# y_test = np.eye(k)[y_test]
# X_train, X_vali = X[:45000], X[45000:]
# y_train, y_vali = y[:45000], y[45000:]
# use_aug = True

#%% Load MNIST data
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

#%% Functions
def plot_result_std_cr_gammax(arrays, gamma_range, title, ylabel, filename):
    """
    arrays: num_rep * num_copyrates * 4
    standard deviation
    skill levels as x axis
    """
    avg = arrays.mean(axis=0)
    std = arrays.std(axis=0)
    lower = std
    upper= std
    plt.errorbar(x=gamma_range, y=avg[:, 0], yerr=[lower[:, 0], upper[:, 0]], label="reweighting both", fmt='-o')
    plt.errorbar(x=gamma_range, y=avg[:, 1], yerr=[lower[:, 1], upper[:, 1]], label="label count reweighting only", fmt='-o')
    plt.errorbar(x=gamma_range, y=avg[:, 2], yerr=[lower[:, 2], upper[:, 2]], label="copy prob reweighting only", fmt='-o')
    plt.errorbar(x=gamma_range, y=avg[:, 3], yerr=[lower[:, 3], upper[:, 3]], label="no reweighting", fmt='-o')
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
    m = 5 # number of users
    gamma_range = np.arange(0.30, 0.501, 0.05)
    repeat = 4 # redundancy
    valid_range = np.arange(50000)
    num_busy = 1 # 1 by default, starting from No.0
    copy_rates = np.zeros(m)
    copy_ids = np.arange(1, 3) # user no.1 is the copycat
    copy_rate = 0.5
    copy_rates[copy_ids] = copy_rate
    num_rep = 5 # repetition
    test_accs = np.zeros((num_rep, len(gamma_range), 4)) # four algorithms to compare
    conf_errors = np.zeros((num_rep, len(gamma_range), 4))
    cp_errors = np.zeros((num_rep, len(gamma_range), 4))
    
    title = "Redundancy:{}, copy probability: {}".format(repeat, copy_rate)
    filename = "Copyrate_layer_lossreweight_r_{}_c_{}_ablation".format(repeat, copy_rate).replace('.', '')
    
    for rep in range(num_rep):
        print("Repetition: {}".format(rep))
        for i, gamma in enumerate(gamma_range):
            print("----------------")
            print("Skill level: {}".format(gamma))
            conf = generate_conf_pairflipper(m, k, gamma)
            labels_train, _, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
            labels_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)
            
            # 1. Reweighting according to both label counts and estimated copy probs
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, use_pretrained=False, model=None, use_aug=False, est_cr=True, reweight="BOTH")
            test_accs[rep, i, 0] = test_acc
            conf_errors[rep, i, 0] = conf_error
            cp_errors[rep, i, 0] = cp_error
            print(est_copyrates[1:])
            plot_conf_mat(est_conf, conf)
            
            print("--------")
            # 2. Reweighing according to label counts only
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, use_pretrained=False, model=None, use_aug=False, est_cr=True, reweight="CNT")
            test_accs[rep, i, 1] = test_acc
            conf_errors[rep, i, 1] = conf_error
            cp_errors[rep, i, 1] = cp_error
            
            print("--------")
            # 3. Reweghting according to estimated copy probs only
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, use_pretrained=False, model=None, use_aug=False, est_cr=True, reweight="CP")
            test_accs[rep, i, 2] = test_acc
            conf_errors[rep, i, 2] = conf_error
            cp_errors[rep, i, 2] = cp_error
            
            print("--------")
            # 4. No reweighting
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, use_pretrained=False, model=None, use_aug=False, est_cr=True, reweight=False)
            test_accs[rep, i, 3] = test_acc
            conf_errors[rep, i, 3] = conf_error
            cp_errors[rep, i, 3] = cp_error
      
            
    plot_result_std_cr_gammax(test_accs, gamma_range, title=title, ylabel="Test accuracy", filename=filename+"_testacc")
    plot_result_std_cr_gammax(cp_errors, gamma_range, title=title, ylabel="Copy probability estimation error", filename=filename+"_cperror")
    plot_result_std_cr_gammax(conf_errors, gamma_range, title=title, ylabel="Confusion matrix estimation error", filename=filename+"_conferror")
