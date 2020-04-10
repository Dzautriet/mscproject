# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:05:08 2020

@author: Zhe Cao
"""

import numpy as np
import matplotlib.pyplot as plt
import gc
from workers import *
from utils import correct_rate, plot_conf_mat
from MBEM import majority_vote, posterior_distribution, posterior_distribution_2
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

#%% First round
m = 10 # number of workers(users) including busy and copycats
gamma_b = 0.3 # worker quality of busy users
gamma_c = 0.6 # worker quality of other users
repeat = 10
valid_range = np.arange(1000)
num_busy = 1 # 1 by default
id_busy = 0 # 0 by default
copy_rates = np.zeros(m)
copy_rates[1:] = 0.4

conf_b = generate_conf_pairflipper(num_busy, k, gamma_b)
conf = generate_conf_pairflipper(m, k, gamma_c)
conf[id_busy] = conf_b

response, workers_train_label, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
response_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)

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

# Estimating true label distribution with incorporating copy rates
est_q_2, est_label_posterior_2, est_conf_2, est_copy_rates = posterior_distribution_2(response, pred_train, workers_on_example, id_busy)
est_q_vali_2, est_label_posterior_vali_2, _, __ = posterior_distribution_2(response_vali, pred_vali, workers_on_example_vali, id_busy)
print("MBEM labels matched (copy rates incorporated) : {:.4f}".format(correct_rate(y_train[valid_range], est_label_posterior_2)))

# # Train
# pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug)
# pred_train_2, pred_vali_2, vali_acc_2, test_acc_2, model_2 = call_train(X_train, valid_range, est_label_posterior_2, X_vali, est_label_posterior_vali_2, y_vali, X_test, y_test, use_aug=use_aug)
# est_q, _, est_conf = posterior_distribution(response, pred_train, workers_on_example)
# plot_conf_mat(est_conf, conf)
# est_q_2, _, est_conf_2, __ = posterior_distribution_2(response, pred_train_2, workers_on_example, id_busy)
# plot_conf_mat(est_conf_2, conf)

#%% Second MBEM iteration
# # MBEM
# est_q, est_label_posterior, est_conf = posterior_distribution(response, pred_train, workers_on_example)
# est_q_vali, est_label_posterior_vali, _ = posterior_distribution(response_vali, pred_vali, workers_on_example_vali)
# print("MBEM labels matched: {:.4f}".format(correct_rate(y_train[valid_range], est_label_posterior)))
# # Modified MBEM
# est_q_2, est_label_posterior_2, est_conf_2, est_copy_rates = posterior_distribution_2(response, pred_train_2, workers_on_example, id_busy)
# est_q_vali_2, est_label_posterior_vali_2, _, __ = posterior_distribution_2(response_vali, pred_vali_2, workers_on_example_vali, id_busy)
# print("MBEM labels matched (copy rates incorporated) : {:.4f}".format(correct_rate(y_train[valid_range], est_label_posterior_2)))
# # Train
# pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug)
# pred_train_2, pred_vali_2, vali_acc_2, test_acc_2, model_2 = call_train(X_train, valid_range, est_label_posterior_2, X_vali, est_label_posterior_vali_2, y_vali, X_test, y_test, use_aug=use_aug)

# est_q, _, est_conf = posterior_distribution(response, pred_train, workers_on_example)
# plot_conf_mat(est_conf, conf)
# est_q_2, _, est_conf_2, __ = posterior_distribution_2(response, pred_train_2, workers_on_example, id_busy)
# plot_conf_mat(est_conf_2, conf)


