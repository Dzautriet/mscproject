# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 19:14:45 2020

@author: Zhe Cao
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

#%% Plot function    
def plot_acc_mul(vali_accs, copy_rate_range, num_iters, title, filename):
    avg = vali_accs.mean(axis=0)
    min_acc = vali_accs.min(axis=0)
    max_acc = vali_accs.max(axis=0)
    lower = avg - min_acc
    upper= max_acc - avg
    plt.errorbar(x=copy_rate_range, y=avg[:, 0], yerr=[lower[:, 0], upper[:, 0]], label="Majority vote", fmt='-o')
    for i in range(1, num_iters+1):
        plt.errorbar(x=copy_rate_range, y=avg[:, i], yerr=[lower[:, i], upper[:, i]], label="Iteration {}".format(i), fmt='-o')
    plt.ylim(min(0.95, vali_accs.min()-0.01), 1.01)
    plt.title(title)
    plt.xlabel("Copy probability")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
#%% Impact of copy rates
m = 4
gamma_b = 0.2
gamma_c = 1.0
repeat = 2
valid_range = np.arange(10000)
num_busy = 1 # 1 by default
id_busy = 0 # 0 by default
copy_rates = np.zeros(m)
copy_rate_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
copy_id = np.arange(1, 2)
num_rep = 5 # repetition
num_iters = 3
vali_accs = np.zeros((num_rep, len(copy_rate_range), num_iters+1))

for rep in range(num_rep):
    print("Repetition: {}".format(rep))
    for i, copy_rate in enumerate(copy_rate_range):
        print("copy prob: {}".format(copy_rate))
        copy_rates[copy_id] = copy_rate
        conf_b = generate_conf_pairflipper(2, k, gamma_b)
        conf = generate_conf_pairflipper(m, k, gamma_c)
        conf[:2] = conf_b
        
        response, workers_train_label, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
        response_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)
        
        # Weighted majority vote
        y_train_wmv = np.sum(response, axis=1) / repeat
        y_vali_corrupt = np.sum(response_vali, axis=1) / repeat
        print("Weighted majority vote labels matched: {:.4f}".format(correct_rate(y_train[valid_range], y_train_wmv)))
        pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug)
        vali_accs[rep, i, 0] = vali_acc
        
        for j in range(num_iters):
            print("Iter {}".format(j))
            est_q, est_label_posterior, est_conf, est_copy_rates = posterior_distribution_2(response, pred_train, workers_on_example, id_busy, copy_rates)
            est_q_vali, est_label_posterior_vali, _, __ = posterior_distribution_2(response_vali, pred_vali, workers_on_example_vali, id_busy, copy_rates)
            # est_q, est_label_posterior, est_conf = posterior_distribution(response, pred_train, workers_on_example)
            # est_q_vali, est_label_posterior_vali, _ = posterior_distribution(response_vali, pred_vali, workers_on_example_vali)
            print("MBEM labels matched: {:.4f}".format(correct_rate(y_train[valid_range], est_label_posterior)))
            # Train
            pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug)
            vali_accs[rep, i, j+1] = vali_acc

filename = "MNIST PWF COPY MBEM_m-2"
plot_acc_mul(vali_accs, copy_rate_range, num_iters, "Pairwise flipper w/ modified MBEM-2", filename+".png")
np.save(filename+".npy", vali_accs)

