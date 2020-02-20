# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:43:37 2020

@author: Zhe Cao
"""

import numpy as np
import matplotlib.pyplot as plt
import gc
from workers import *
from utils import correct_rate, plot_conf_mat
from MBEM import majority_vote, posterior_distribution
from train import call_train

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

#%% Baseline
# _, __, vali_acc, test_acc, model = call_train(X_train, y_train, X_vali, y_vali, y_vali, X_test, y_test, use_aug=use_aug)

#%% Traning round 1
m = 1 # number of workers(experts)
k = 10 # number of classes
gamma = 1.0 # experts are expected to have higher qualities
# class_wise = True
repeat = 1
valid_range = np.arange(1000)

conf = generate_conf_pairflipper(m, k, gamma)
response, workers_train_label, workers_on_example = generate_labels_weight(y_train[valid_range], repeat, conf)
response_vali, _, workers_on_example_vali = generate_labels_weight(y_vali, repeat, conf) # Also generate noisy labels for validation set, which will be used during training

# Majority vote
y_train_r1 = majority_vote(response)
y_vali_r1 = majority_vote(response_vali)
print("Majority vote labels matched: {:.4f}".format(correct_rate(y_train[valid_range], y_train_r1)))
# Training
pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, y_train_r1, X_vali, y_vali_r1, y_vali, X_test, y_test, use_aug=use_aug)

#%% Users provided some feedback
m = 3 # number of workers(users)
gamma = 0.4
# class_wise=True
repeat = m
valid_range = np.arange(50000)
# sleep_rates = np.zeros(m)
# sleep_rates[np.random.choice(m, 2)] = 0.5
copy_rates = np.zeros(m)
copy_rates[2] = 0.6

conf = generate_conf_pairflipper(m, k, gamma)
response, workers_train_label, workers_on_example = generate_labels_weight_copycat(y_train[valid_range], repeat, conf, pred_train, copy_rates)
response_vali, _, workers_on_example_vali = generate_labels_weight_copycat(y_vali, repeat, conf, pred_vali, copy_rates)

# Weighted majority vote
y_train_wmv = np.sum(response, axis=1) / repeat
y_vali_corrupt = np.sum(response_vali, axis=1) / repeat
print("Weighted majority vote labels matched: {:.4f}".format(correct_rate(y_train[valid_range], y_train_wmv)))
pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug)

#%% MBEM
# pred_train_wmv is used as the initial posterior distribution
est_q, est_label_posterior, est_conf = posterior_distribution(response, pred_train, workers_on_example)
est_q_vali, est_label_posterior_vali, _ = posterior_distribution(response_vali, pred_vali, workers_on_example_vali)
print("MBEM labels matched: {:.4f}".format(correct_rate(y_train[valid_range], est_label_posterior)))
pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug)

est_q, __, est_conf = posterior_distribution(response, pred_train, workers_on_example)
plot_conf_mat(est_conf, conf)
# print("Sleep rates:", sleep_rates)
print("Copy rates:", copy_rates)
print("Estimated q:", est_q)
print("Real marginal:", np.mean(y_train[valid_range], axis=0))


