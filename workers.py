# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:52:33 2020

@author: Zhe Cao
"""

import numpy as np

def generate_conf_hammerspammer(m,k,gamma,class_wise):
    """
    Arguments:
    m: number of workers
    k: number of classes
    gamma: worker quality
    class_wise: boolean

    Return:
    conf: confusion matrix, m * k * k, row for true classes, column for customer labels
    """
    conf = np.ones((m,k,k)) / k
    for i in range(m): 
        # non-classwise hammer-spammer distribution
        if not class_wise:
            if np.random.random() < gamma:
                conf[i] = np.identity(k)
            # To avoid numerical issues changing the spammer matrix each element slightly    
            else:
                conf[i] = conf[i] + 0.01*np.identity(k)
                conf[i] = conf[i] / np.sum(conf[i]) * k     
        else:
            # classwise hammer-spammer distribution    
            for j in range(k):
                if np.random.random() < gamma:
                    conf[i,j,:] = 0
                    conf[i,j,j] = 1 
                # again to avoid numerical issues changing the spammer distribution slightly 
                # by generating uniform random variable between 0.1 and 0.11
                else:
                    conf[i,j,:] = 1
                    conf[i,j,j] = 1 + np.random.uniform(0.1,0.11)
                    conf[i,j,:] = conf[i,j,:] / np.sum(conf[i,j,:])
    return conf

def generate_conf_pairflipper(m, k, gamma):
    """
    Arguments:
    m: number of workers
    k: number of classes
    gamma: worker quality

    Return:
    conf: confusion matrix, m * k * k, row for true classes, column for customer labels
    """
    conf = np.zeros((m, k, k))
    row_id = np.arange(k)
    col_id = np.arange(k)
    for i in range(m):
        diag = np.random.normal(gamma, 0.01)
        conf_m = np.eye(k) * diag
        np.random.shuffle(col_id)
        conf_m[row_id, col_id] += (1-diag)
        conf[i, :, :] = conf_m
    return conf

def generate_labels_weight(y, repeat, conf):
    """
    Arguments:
    y: training set
    repeat: redundancy
    """
    n = y.shape[0]
    m, k = conf.shape[0], conf.shape[1]

    workers_train_label = {}
    for i in range(repeat):
        workers_train_label['softmax_' + str(i) + '_label'] = np.zeros((n, k))

    response = np.zeros((n, m, k))
    workers_on_example = np.zeros((n, repeat), dtype=int) # workers ID per sample

    for i in range(n):
        # Randomly select #repeat workers
        workers_on_example[i] = np.sort(np.random.choice(m, repeat, replace=False)) # One worker can only label on sample once
        for count, worker_id in enumerate(workers_on_example[i]):
            corrupt_label = np.random.multinomial(1, conf[worker_id, np.argmax(y[i]), :])
            assert corrupt_label.sum() == 1
            response[i, worker_id, :] = corrupt_label
            workers_train_label['softmax_' + str(count) + '_label'][i] = corrupt_label

    return response, workers_train_label, workers_on_example

def generate_labels_weight_sleepy(y, repeat, conf, y_r1, sleep_rates):
    """
    Arguments:
    y: training set labels
    repeat: redundancy
    y_r1: training set labels in round one
    sleep rates: sleep rate of each user
    """
    n = y.shape[0]
    m, k = conf.shape[0], conf.shape[1]

    workers_train_label = {}
    for i in range(repeat):
        workers_train_label['softmax_' + str(i) + '_label'] = np.zeros((n, k))

    response = np.zeros((n, m, k))
    workers_on_example = np.zeros((n, repeat), dtype=int) # workers ID per sample

    for i in range(n):
        # Randomly select #repeat workers
        workers_on_example[i] = np.sort(np.random.choice(m, repeat, replace=False)) # One worker can only label on sample once
        for count, worker_id in enumerate(workers_on_example[i]):
            if np.random.random() < sleep_rates[worker_id]:
                # User sleeps, giving the same label as round one
                corrupt_label = y_r1[i]
            else:
                corrupt_label = np.random.multinomial(1, conf[worker_id, np.argmax(y[i]), :])
                assert corrupt_label.sum() == 1
            response[i, worker_id, :] = corrupt_label
            workers_train_label['softmax_' + str(count) + '_label'][i] = corrupt_label

    return response, workers_train_label, workers_on_example