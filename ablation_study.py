# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 19:14:30 2020

@author: Zhe Cao

Ablation study
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
import argparse

def plot_result_std_cr(arrays, copy_rate_range, title, ylabel, filename):
    """
    arrays: num_rep * num_copyrates * 6
    standard deviation
    """
    avg = arrays.mean(axis=0)
    std = arrays.std(axis=0)
    lower = std
    upper= std
    plt.errorbar(x=copy_rate_range, y=avg[:, 0], yerr=[lower[:, 0], upper[:, 0]], capsize=4, label="full model", fmt='--o')
    plt.errorbar(x=copy_rate_range, y=avg[:, 1], yerr=[lower[:, 1], upper[:, 1]], capsize=4, label="w/o skill level reweight", fmt='--o')
    plt.errorbar(x=copy_rate_range, y=avg[:, 2], yerr=[lower[:, 2], upper[:, 2]], capsize=4, label="w/o abstention rate reweight", fmt='--o')
    plt.errorbar(x=copy_rate_range, y=avg[:, 3], yerr=[lower[:, 3], upper[:, 3]], capsize=4, label="w/o label count reweight", fmt='--o')
    # plt.errorbar(x=copy_rate_range, y=avg[:, 4], yerr=[lower[:, 4], upper[:, 4]], capsize=4, label="w/o 2-stage grad update", fmt='--o')
    plt.errorbar(x=copy_rate_range, y=avg[:, 5], yerr=[lower[:, 5], upper[:, 5]], capsize=4, label="w/o pretraining", fmt='--o')
    plt.ylim(0.0, arrays.max()+0.01)
    plt.title(title)
    plt.xlabel("Copy probability")
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(filename+".png", bbox_inches='tight')
    np.save(filename+".npy", arrays)
    plt.show()
    
#%% Main
if __name__ == "__main__":
    #%% Parse arguments
#     parser = argparse.ArgumentParser(description='Process some integers.')
#     parser.add_argument('gamma_b', type=float, help='skill level of the busy user')
#     parser.add_argument('gamma_c', type=float, help='skill level of the other users')
#     parser.add_argument('dataset', type=str, help='mnist or cifar10')
#     args = parser.parse_args()
    
#     dataset = args.dataset
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
#     gamma_b = args.gamma_b
#     gamma_c = args.gamma_c
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
    test_accs = np.zeros((num_rep, len(copy_rate_range), 6)) # six algorithms to compare
    conf_errors = np.zeros((num_rep, len(copy_rate_range), 6))
    cp_errors = np.zeros((num_rep, len(copy_rate_range), 6))
    
    title = "Redundancy:{}, skill level: {} & {}".format(repeat, gamma_b, gamma_c)
    filename = "Gamma_reweight_r_{}_g_{}_{}_1copy_ablation".format(repeat, gamma_b, gamma_c).replace('.', '')
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
            
            # 1. Reweighting according to gamma + both label counts and estimated copy probs
            # Initialise with majority vote
            print("Pretraining model on weighted majority vote...")
            y_train_wmv = np.sum(labels_train, axis=1) / repeat
            y_vali_corrupt = np.sum(labels_vali, axis=1) / repeat
            pred_train, _, _, _, init_model = call_train_mbem(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug, dataset=dataset)
            _, _, est_conf = posterior_distribution(labels_train, pred_train, workers_on_example)
            plot_conf_mat_h(est_conf, conf)
            print("Reweighting: gamma + label counts + copy probs")
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=True, model=init_model, conf_init=None, use_aug=use_aug, est_cr=True, reweight="GAMMA+BOTH")
            test_accs[rep, i, 0] = test_acc
            conf_errors[rep, i, 0] = conf_error
            cp_errors[rep, i, 0] = cp_error
            print(est_copyrates[1:])
            plot_conf_mat_h(est_conf, conf)
            # raise Exception()
            
            # 2. Reweighting according to both label counts and estimated copy probs
            print("Ablating: gamma reweighting")
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=True, model=init_model, use_aug=use_aug, est_cr=True, reweight="BOTH")
            test_accs[rep, i, 1] = test_acc
            conf_errors[rep, i, 1] = conf_error
            cp_errors[rep, i, 1] = cp_error
            print(est_copyrates[1:])
            plot_conf_mat_h(est_conf, conf)
            
            # 3. Reweighting according to gamma + label counts
            print("Ablating: copy probs")
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=True, model=init_model, conf_init=None, use_aug=use_aug, est_cr=True, reweight="GAMMA+CNT")
            test_accs[rep, i, 2] = test_acc
            conf_errors[rep, i, 2] = conf_error
            cp_errors[rep, i, 2] = cp_error
            print(est_copyrates[1:])
            plot_conf_mat_h(est_conf, conf)
            
            # 4. Reweighting according to gamma + estimated copy probs
            print("Ablating: label counts")
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=True, model=init_model, conf_init=None, use_aug=use_aug, est_cr=True, reweight="GAMMA+CP")
            test_accs[rep, i, 3] = test_acc
            conf_errors[rep, i, 3] = conf_error
            cp_errors[rep, i, 3] = cp_error
            print(est_copyrates[1:])
            plot_conf_mat_h(est_conf, conf)
            
            # # 5. Two stages
            # print("Ablating: two-stage")
            # est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
            #                                                 conf, copy_rates, two_stage=True, use_pretrained=True, model=init_model, conf_init=None, use_aug=use_aug, est_cr=True, reweight="GAMMA+BOTH")
            # test_accs[rep, i, 4] = test_acc
            # conf_errors[rep, i, 4] = conf_error
            # cp_errors[rep, i, 4] = cp_error
            # print(est_copyrates[1:])
            # plot_conf_mat_h(est_conf, conf)
            
            # 6. Train from scratch
            print("Ablating: pretraining on weighted majority votes")
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=False, model=None, conf_init=None, use_aug=use_aug, est_cr=True, reweight="GAMMA+BOTH")
            test_accs[rep, i, 5] = test_acc
            conf_errors[rep, i, 5] = conf_error
            cp_errors[rep, i, 5] = cp_error
            print(est_copyrates[1:])
            plot_conf_mat_h(est_conf, conf)
            
    plot_result_std_cr(test_accs, copy_rate_range, title=title, ylabel="Test accuracy", filename=filename+"_testacc")
    plot_result_std_cr(cp_errors, copy_rate_range, title=title, ylabel="Copy probability estimation error", filename=filename+"_cperror")
    plot_result_std_cr(conf_errors, copy_rate_range, title=title, ylabel="Confusion matrix estimation error", filename=filename+"_conferror")
    