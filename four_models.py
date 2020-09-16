# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:37:32 2020

@author: Zhe Cao

Copycat scenario: systematic comparison (4 algorithms)
Reweighting mechanism
Core of ablation study
Gamma reweighting added
Pretraining on weighted majority vote
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
from mbem import posterior_distribution
from train import call_train
import models
import gc
import argparse

#%% Functions & classes    
def plot_result_std_cr(arrays, copy_rate_range, title, ylabel, filename):
    """
    arrays: num_rep * num_copyrates * 4
    standard deviation
    """
    avg = arrays.mean(axis=0)
    std = arrays.std(axis=0)
    lower = std
    upper= std
    plt.errorbar(x=copy_rate_range, y=avg[:, 0], yerr=[lower[:, 0], upper[:, 0]], capsize=4, label="proposed method", fmt='--o', color=u'#1f77b4')
    if np.any(avg[:, 1] != 0):
        plt.errorbar(x=copy_rate_range, y=avg[:, 1], yerr=[lower[:, 1], upper[:, 1]], capsize=4, label="vanilla", fmt='--o', color=u'#ff7f0e')
    if np.any(avg[:, 2] != 0):
        plt.errorbar(x=copy_rate_range, y=avg[:, 2], yerr=[lower[:, 2], upper[:, 2]], capsize=4, label="majority vote", fmt='--o', color=u'#2ca02c')
    if np.any(avg[:, 3] != 0):
        plt.errorbar(x=copy_rate_range, y=avg[:, 3], yerr=[lower[:, 3], upper[:, 3]], capsize=4, label="MBEM", fmt='--o', color=u'#d62728')
    plt.ylim(0.0, arrays.max()+0.01)
    plt.title(title)
    plt.xlabel("Copy probability")
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(filename+".png", bbox_inches='tight')
    np.save(filename+".npy", arrays)
    # plt.show()
    plt.close()

#%% Main
if __name__ == "__main__":
    #%% Parse arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('gamma_b', type=float, help='skill level of the busy user')
    parser.add_argument('gamma_c', type=float, help='skill level of the other users')
    parser.add_argument('dataset', type=str, help='mnist or cifar10')
    args = parser.parse_args()
    
    dataset = args.dataset
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
    # gamma_b = .35 # skill level of the busy user
    # gamma_c = .35 # skill level of the other users
    gamma_b = args.gamma_b
    gamma_c = args.gamma_c
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
    num_rep = 3 # repetition
    mbem_round = 1
    test_accs = np.zeros((num_rep, len(copy_rate_range), 4)) # four algorithms to compare
    conf_errors = np.zeros((num_rep, len(copy_rate_range), 4))
    cp_errors = np.zeros((num_rep, len(copy_rate_range), 4))
    
    title = "Redundancy:{}, skill level: {} & {}".format(repeat, gamma_b, gamma_c)
    filename = "{}_Gamma_reweight_r_{}_g_{}_{}_4comp_1copy_1stage_full".format(dataset, repeat, gamma_b, gamma_c).replace('.', '')
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    filename = '/'.join(['.', result_dir, filename])
    
    for rep in range(num_rep):
        print("Repetition: {}".format(rep))
        for i, copy_rate_value in enumerate(copy_rate_range):
            copy_rates[copy_ids] = copy_rate_value
            print("----------------")
            print("Copy rates: {}".format(copy_rates))
            conf_b = generate_conf_pairflipper(num_busy, k, gamma_b)
            conf = generate_conf_pairflipper(m, k, gamma_c)
            conf[:num_busy] = conf_b
            labels_train, _, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
            labels_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)
            
            # 1. pretrain on majority vote
            print("Pretraining model on weighted majority vote...")
            y_train_wmv = np.sum(labels_train, axis=1) / repeat
            y_vali_corrupt = np.sum(labels_vali, axis=1) / repeat
            pred_train, pred_vali, vali_acc, test_acc, init_model = call_train_mbem(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug, dataset=dataset)
            test_accs[rep, i, 2] = test_acc
            # conf_errors[rep, i, 2] = conf_error # not applicable
            # cp_errors[rep, i, 2] = cp_error # not applicable
            # plot_conf_mat_h(est_conf, conf)

            # 2. Reweighting according to gamma + both label counts and estimated copy probs
            print("--------")
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=True, model=init_model, conf_init=None, use_aug=use_aug, est_cr=True, reweight="GAMMA+BOTH", dataset=dataset)
            test_accs[rep, i, 0] = test_acc
            conf_errors[rep, i, 0] = conf_error
            cp_errors[rep, i, 0] = cp_error
            print(est_copyrates[1:])
            # plot_conf_mat_h(est_conf, conf)
            
            # 3. w/o copy rate est.
            print("--------")
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=False, model=None, use_aug=use_aug, est_cr=False, reweight=False, dataset=dataset)
            test_accs[rep, i, 1] = test_acc
            conf_errors[rep, i, 1] = conf_error
            # cp_errors[rep, i, 1] = cp_error # not applicable
            
            # 4. MBEM
            print("--------")
            for j in range(mbem_round):
                est_q, est_label_posterior, est_conf = posterior_distribution(labels_train, pred_train, workers_on_example)
                est_q_vali, est_label_posterior_vali, _ = posterior_distribution(labels_vali, pred_vali, workers_on_example_vali)
                # Train
                pred_train, pred_vali, vali_acc, test_acc, model = call_train_mbem(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug, dataset=dataset)
            conf_error = np.mean([np.linalg.norm(est_conf[i] - conf[i]) for i in range(m)])
            test_accs[rep, i, 3] = test_acc
            conf_errors[rep, i, 3] = conf_error
            # cp_errors[rep, i, 3] = cp_error # not applicable
            # plot_conf_mat_h(est_conf, conf)
            
            
    plot_result_std_cr(test_accs, copy_rate_range, title=title, ylabel="Test accuracy", filename=filename+"_testacc")
    plot_result_std_cr(cp_errors, copy_rate_range, title=title, ylabel="Copy probability estimation error", filename=filename+"_cperror")
    plot_result_std_cr(conf_errors, copy_rate_range, title=title, ylabel="Confusion matrix estimation error", filename=filename+"_conferror")
