# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 15:56:21 2020

@author: Zhe Cao

Confusion matrix evolution illustration
Varied skill levels
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
    gamma_b = .4 # skill level of the busy user
    gamma_c = .4 # skill level of the other users
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
    copy_rate_range = np.arange(0.0, 1.01, 0.25)
    
    conf_collection = np.zeros((len(copy_rate_range)+1, m, k, k))
    
    filename = "Conf_evo_r_{}_g_{}_{}".format(repeat, gamma_b, gamma_c).replace('.', '')
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    filename = '/'.join(['.', result_dir, filename])    
    
    conf_b = generate_conf_pairflipper(num_busy, k, gamma_b)
    conf = generate_conf_pairflipper(m, k, gamma_c)
    conf[:num_busy] = conf_b
    conf_collection[0] = conf

    for i, copy_rate_value in enumerate(copy_rate_range):
        copy_rates[copy_ids] = copy_rate_value
        print("----------------")
        print("Copy rates: {}".format(copy_rates))
        
        labels_train, _, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
        labels_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)
        
        # 1. Reweighting according to gamma + both label counts and estimated copy probs
        # Initialise with majority vote
        print("Pretraining model on weighted majority vote...")
        y_train_wmv = np.sum(labels_train, axis=1) / repeat
        y_vali_corrupt = np.sum(labels_vali, axis=1) / repeat
        print("Reweighting: gamma + label counts + copy probs")
        est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                        conf, copy_rates, two_stage=False, use_pretrained=False, model=None, conf_init=None, use_aug=use_aug, est_cr=True, reweight="GAMMA+BOTH")
        print(est_copyrates[1:])
        conf_collection[i+1] = est_conf
        plot_conf_mat_h(est_conf, conf)
        
    np.save(filename+".npy", conf_collection)