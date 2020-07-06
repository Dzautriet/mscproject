# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:27:58 2020

@author: Zhe Cao

Reweighting: adaptive
"""

import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.datasets import make_blobs
from workers import *
from utils import correct_rate, plot_conf_mat
from MBEM import majority_vote, posterior_distribution, posterior_distribution_2
from train import call_train, call_train_blobs

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
def filtercopycat(response, workers_on_example, copy_ids):
    '''Filter out copycat response'''
    filtered_response = response.copy()
    filtered_workers_on_example = workers_on_example.copy()
    filtered_response[:, copy_ids, :] = 0
    for copy_id in copy_ids:
        filtered_workers_on_example[filtered_workers_on_example==copy_id] = -1 # fill copycat with -1
    return filtered_response, filtered_workers_on_example

def weight_func(est_match_rate, quality_factor):
    weight = 1 / (np.exp(5*(est_match_rate-quality_factor))+1)
    # weight[:] = 1.
    # weight[1] = 0.35
    return weight.reshape(1, -1, 1)

def estcopycat(response, response_vali):
    response_concat = np.concatenate((response, response_vali), axis=0)
    n, m, k = response_concat.shape
    est_match_rate = np.ones(m)
    for i in range(1, m):
        est_match_rate[i] = np.all(response_concat[:, 0] == response_concat[:, i], axis=1).sum() / response_concat[:, i].sum()
    if len(est_match_rate[est_match_rate < 0.5]) > 0:
        est_match_rate[0] = est_match_rate[est_match_rate < 0.5].mean()
    else:
        est_match_rate[0] = 0.5
    return est_match_rate

def plot_acc_t(vali_accs, num_iters, title, filename):
    avg = vali_accs.mean(axis=0)
    min_acc = vali_accs.min(axis=0)
    max_acc = vali_accs.max(axis=0)
    lower = avg - min_acc
    upper= max_acc - avg
    plt.errorbar(x=range(num_iters+1), y=avg[0, :], yerr=[lower[0, :], upper[0, :]], label="Original MBEM", fmt='-o')
    plt.errorbar(x=range(num_iters+1), y=avg[1, :], yerr=[lower[1, :], upper[1, :]], label="Copycat response removed", fmt='-o')
    plt.errorbar(x=range(num_iters+1), y=avg[2, :], yerr=[lower[2, :], upper[2, :]], label="Copycat response reweighted", fmt='-o')
    plt.xticks(range(num_iters+1))
    plt.ylim(min(0.95, vali_accs.min()-0.01), 1.01)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # plt.savefig(filename, bbox_inches='tight')
    plt.show()

#%% Filter out copycat's reponse
m = 5
gamma = .35
gamma_2 = .35
repeat = 2
valid_range = np.arange(50000)
num_busy = 1 # 1 by default
id_busy = 0 # 0 by default
copy_rates = np.zeros(m)
copy_ids = np.arange(1, 2)
copy_rates[copy_ids] = .7                             
num_rep = 1 # repetition
num_iters = 2

vali_accs = np.zeros((num_rep, 3, num_iters+1))

for rep in range(num_rep):
    print("Repetition: {}".format(rep))
    conf_2 = generate_conf_pairflipper(1, k, gamma_2)
    conf = generate_conf_pairflipper(m, k, gamma)
    conf[1:2] = conf_2
    
    response, _, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
    response_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)

    # 1
    y_train_wmv = np.sum(response, axis=1) / repeat
    y_vali_corrupt = np.sum(response_vali, axis=1) / repeat
    print("Weighted majority vote labels matched: {:.4f}".format(correct_rate(y_train[valid_range], y_train_wmv)))
    pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug)
    vali_accs[rep, 0, 0] = vali_acc
    for j in range(num_iters):
        print("Iter {}".format(j))
        # est_q, est_label_posterior, est_conf, est_copy_rates = posterior_distribution_2(response, pred_train, workers_on_example, id_busy, copy_rates)
        # est_q_vali, est_label_posterior_vali, _, __ = posterior_distribution_2(response_vali, pred_vali, workers_on_example_vali, id_busy, copy_rates)
        est_q, est_label_posterior, est_conf = posterior_distribution(response, pred_train, workers_on_example)
        est_q_vali, est_label_posterior_vali, _ = posterior_distribution(response_vali, pred_vali, workers_on_example_vali)
        print("MBEM labels matched: {:.4f}".format(correct_rate(y_train[valid_range], est_label_posterior)))
        # Train
        pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug)
        vali_accs[rep, 0, j+1] = vali_acc
    
    # 2
    filtered_response, filtered_workers_on_example = filtercopycat(response, workers_on_example, copy_ids)
    filtered_response_vali, filtered_workers_on_example_vali = filtercopycat(response_vali, workers_on_example_vali, copy_ids)
    y_train_wmv = np.sum(filtered_response, axis=1)
    y_train_wmv /= repeat
    y_vali_corrupt = np.sum(filtered_response_vali, axis=1)
    y_vali_corrupt /= repeat
    print("Weighted majority vote labels matched: {:.4f}".format(correct_rate(y_train[valid_range], y_train_wmv)))
    pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug)
    vali_accs[rep, 1, 0] = vali_acc
    for j in range(num_iters):
        print("Iter {}".format(j))
        est_q, est_label_posterior, est_conf = posterior_distribution(filtered_response, pred_train,filtered_workers_on_example)
        est_q_vali, est_label_posterior_vali, _ = posterior_distribution(filtered_response_vali, pred_vali, filtered_workers_on_example_vali)
        print("MBEM labels matched: {:.4f}".format(correct_rate(y_train[valid_range], est_label_posterior)))
        # Train
        pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug)
        vali_accs[rep, 1, j+1] = vali_acc
        
    # 3
    est_match_rate = estcopycat(response, response_vali)
    quality_factor = 0.5
    weight = weight_func(est_match_rate, quality_factor)
    est_response = response.copy() * weight
    est_response_vali = response_vali.copy() * weight
    y_train_wmv = np.sum(est_response, axis=1)
    y_train_wmv /= repeat
    y_vali_corrupt = np.sum(est_response_vali, axis=1)
    y_vali_corrupt /= repeat
    print("Weighted majority vote labels matched: {:.4f}".format(correct_rate(y_train[valid_range], y_train_wmv)))
    pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug)
    vali_accs[rep, 2, 0] = vali_acc
    for j in range(num_iters):
        print("Iter {}".format(j))
        quality_factor = np.diagonal(est_conf[0]).mean()
        weight = weight_func(est_match_rate, quality_factor)
        est_response = response.copy() * weight
        est_response_vali = response_vali.copy() * weight
        est_q, est_label_posterior, est_conf = posterior_distribution(est_response, pred_train, workers_on_example)
        est_q_vali, est_label_posterior_vali, _ = posterior_distribution(est_response_vali, pred_vali, workers_on_example_vali)
        print("MBEM labels matched: {:.4f}".format(correct_rate(y_train[valid_range], est_label_posterior)))
        # Train
        pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug)
        vali_accs[rep, 2, j+1] = vali_acc

filename = "MNIST PWF COPY REWEIGHT-A 06"
plot_acc_t(vali_accs, num_iters, "Pairwise flipper w/ copycat response reweighted adaptive 06", filename+".png")
# np.save(filename+".npy", vali_accs)


