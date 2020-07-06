# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:54:02 2020

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

#%% Generate Gaussian blobs data
# k = 2
# n_features = 2
# X, y = make_blobs(n_samples=12000, centers=k, n_features=n_features, random_state=42)
# y = np.eye(k)[y]
# X_train, X_vali, X_test = X[:10000], X[10000:11000], X[11000:]
# y_train, y_vali, y_test = y[:10000], y[10000:11000], y[11000:]

#%% Plot function
# def plot_acc(vali_accs, gamma_n_range, num_iters, title):
#     plt.plot(gamma_n_range, vali_accs[:, 0], label="Majority vote")
#     for i in range(1, num_iters+1):
#         plt.plot(gamma_n_range, vali_accs[:, i], label="Iteration {}".format(i))
#     # plt.ylim(min(0.7, vali_accs.min()-0.01), 1.01)
#     plt.title(title)
#     plt.xlabel("Correctness probability")
#     plt.ylabel("Accuracy")
#     plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
#     plt.show()
    
def plot_acc_mul(vali_accs, gamma_n_range, num_iters, title, filename):
    avg = vali_accs.mean(axis=0)
    min_acc = vali_accs.min(axis=0)
    max_acc = vali_accs.max(axis=0)
    lower = avg - min_acc
    upper= max_acc - avg
    plt.errorbar(x=gamma_n_range, y=avg[:, 0], yerr=[lower[:, 0], upper[:, 0]], label="Majority vote", fmt='-o')
    for i in range(1, num_iters+1):
        plt.errorbar(x=gamma_n_range, y=avg[:, i], yerr=[lower[:, i], upper[:, i]], label="Iteration {}".format(i), fmt='-o')
    plt.ylim(min(0.95, vali_accs.min()-0.01), 1.01)
    plt.title(title)
    plt.xlabel("Correctness probability")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
#%% Impact of noise
m = 4 # number of users
gamma_n_range = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0] # worker quality of noisy users (target)
gamma = 1.0 # worker quality of other users
repeat = 1 # redundancy
valid_range = np.arange(10000)
num_n = 2 # number of noisy users
id_n = np.arange(num_n)
num_rep = 5 # repetition
num_iters = 5
vali_accs = np.zeros((num_rep, len(gamma_n_range), num_iters+1))

for rep in range(num_rep):
    print("rep: {}".format(rep))
    for i, gamma_n in enumerate(gamma_n_range):
        print("gamma_n: {}".format(gamma_n))
        # Pair-wise flipper
        conf_b = generate_conf_pairflipper(num_n, k, gamma_n)
        conf = generate_conf_pairflipper(m, k, gamma)
        # Hammer-spammer
        # conf_b = generate_conf_hammerspammer_new(num_n, k, gamma_n)
        # conf = generate_conf_hammerspammer_new(m, k, gamma)
        conf[id_n] = conf_b
        
        response, workers_train_label, workers_on_example = generate_labels_weight(y_train[valid_range], repeat, conf)
        response_vali, _, workers_on_example_vali = generate_labels_weight(y_vali, repeat, conf) # Also generate noisy labels for validation set, which will be used during training
        
        # Weighted majority vote
        y_train_wmv = np.sum(response, axis=1) / repeat
        y_vali_corrupt = np.sum(response_vali, axis=1) / repeat
        print("Weighted majority vote labels matched: {:.4f}".format(correct_rate(y_train[valid_range], y_train_wmv)))
        pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug)
        # pred_train, pred_vali, vali_acc, test_acc, model = call_train_blobs(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test)
        vali_accs[rep, i, 0] = vali_acc
        
        for j in range(num_iters):
            print("Iter {}".format(j))
            est_q, est_label_posterior, est_conf = posterior_distribution(response, pred_train, workers_on_example)
            est_q_vali, est_label_posterior_vali, _ = posterior_distribution(response_vali, pred_vali, workers_on_example_vali)
            print("MBEM labels matched: {:.4f}".format(correct_rate(y_train[valid_range], est_label_posterior)))
            # Train
            pred_train, pred_vali, vali_acc, test_acc, model = call_train(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug)
            # pred_train, pred_vali, vali_acc, test_acc, model = call_train_blobs(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test)
            vali_accs[rep, i, j+1] = vali_acc

filename = "MULTI MNIST PWF 2"
plot_acc_mul(vali_accs, gamma_n_range, num_iters, "{} noisy user(s)".format(num_n), filename+".png")
np.save(filename+".npy", vali_accs)
