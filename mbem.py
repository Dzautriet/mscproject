# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:54:32 2020

@author: Zhe Cao

Reimplemention of MBEM
(Khetan, A., Lipton, Z. C. & Anandkumar, A. (2018), 
Learning from noisy singly-labeled data, 
in ‘International Conference on Learning Representations’.)
(https://github.com/khetan2/MBEM)
"""

import numpy as np

def majority_vote(response):
    """
    Arguments:
    response: n * m * k
    """
    n = response.shape[0]
    k = response.shape[2]
    vote = np.sum(response, axis=1) # n * k
    y_mv = np.zeros((n, k))

    for i in range(n):
        vote_result = np.where(vote[i] == vote[i].max())[0]
        vote_choice = np.random.choice(vote_result) # Break ties
        y_mv[i, vote_choice] = 1
    
    return y_mv

def posterior_distribution(response, pred_train, workers_on_example):
    """
    Computes posterior probability distribution of true labels given noisy 
    label and model prediction
    Arguments:
    response: n * m * k
    pred_train: n * k
    workers_on_example: n * repeat
    """
    n, m, k = response.shape
    repeat = workers_on_example.shape[1]
    est_conf = np.zeros((m, k, k))
    epsilon = 1e-10
    est_label_posterior = np.zeros((n, k))

    # Estimate confusion matrix pi
    for i in range(n):
        for j in workers_on_example[i]:
            if j in range(m): # added to filter out copycat
                est_conf[j, np.argmax(pred_train[i]), np.argmax(response[i, j])] += 1
    # Regularise estimated confusion matrix
    for i in range(m):
        for j in range(k):
            if np.all(est_conf[i, j, :] == 0):
                est_conf[i, j, :] = 1 / k
            else:
                est_conf[i, j, :][est_conf[i, j, :] == 0] = epsilon
            est_conf[i, j, :] = est_conf[i, j, :] / np.sum(est_conf[i, j, :])

    # Estimate marginal distribution q
    est_q = np.sum(pred_train, axis=0)
    est_q /= np.sum(est_q)

    # Estimate posterior distribution P
    for i in range(n):
        for j in workers_on_example[i]:
            if j in range(m): # added to filter out copycat
                est_label_posterior[i] += np.log(np.clip(est_conf[j, :, :] @ response[i, j], 1e-10, 1.-1e-7))
        est_label_posterior[i] = np.exp(est_label_posterior[i]) * est_q
        est_label_posterior[i] /= np.sum(est_label_posterior[i])

    return est_q, est_label_posterior, est_conf

def posterior_distribution_3(response, pred_train, workers_on_example):
    """
    Estimating copy
    Computes posterior probability distribution of true labels given noisy 
    label and model prediction
    Arguments:
    response: n * m * k
    pred_train: n * k
    workers_on_example: n * repeat
    """
    n, m, k = response.shape
    repeat = workers_on_example.shape[1]
    est_conf = np.zeros((m, k, k))
    epsilon = 1e-10
    est_label_posterior = np.zeros((n, k))

    # Estimate confusion matrix pi
    for i in range(n):
        for j in workers_on_example[i]:
            if j in range(m): # added to filter out copycat
                est_conf[j, np.argmax(pred_train[i]), np.argmax(response[i, j])] += 1
    # Regularise estimated confusion matrix
    for i in range(m):
        for j in range(k):
            if np.all(est_conf[i, j, :] == 0):
                est_conf[i, j, :] = 1 / k
            else:
                est_conf[i, j, :][est_conf[i, j, :] == 0] = epsilon
            est_conf[i, j, :] = est_conf[i, j, :] / np.sum(est_conf[i, j, :])

    # Estimate marginal distribution q
    est_q = np.sum(pred_train, axis=0)
    est_q /= np.sum(est_q)

    # Estimate posterior distribution P
    for i in range(n):
        for j in workers_on_example[i]:
            if j in range(m): # added to filter out copycat
                est_label_posterior[i] += np.log(est_conf[j, :, :] @ response[i, j])
        est_label_posterior[i] = np.exp(est_label_posterior[i]) * est_q
        est_label_posterior[i] /= np.sum(est_label_posterior[i])

    return est_q, est_label_posterior, est_conf

def posterior_distribution_2(response, pred_train, workers_on_example, id_busy, copy_rates):
    """
    Reweighting added
    Computes posterior probability distribution of true labels given noisy 
    label and model prediction
    Incorporate copycat rates
    Arguments:
    response: n * m * k
    pred_train: n * k
    workers_on_example: n * repeat
    id_busy: index of the busy user
    """
    n, m, k = response.shape
    repeat = workers_on_example.shape[1]
    est_conf = np.zeros((m, k, k))
    # est_copy_rates = np.zeros(m)
    # est_copy_rates[1:] = 0.4 # pretend we know the true copy rates
    est_copy_rates = copy_rates
    epsilon = 1e-10
    est_label_posterior = np.zeros((n, k))
    uniform_v = np.ones(k) / k
    
    # Estimate confusion matrix pi
    for i in range(n):
        for j in workers_on_example[i]:
            if j in range(m): # added to filter out copycat
                est_conf[j, np.argmax(pred_train[i]), np.argmax(response[i, j])] += 1
    # Regularise estimated confusion matrix
    for i in range(m):
        for j in range(k):
            if np.all(est_conf[i, j, :] == 0):
                est_conf[i, j, :] = 1 / k
            else:
                est_conf[i, j, :][est_conf[i, j, :] == 0] = epsilon
            est_conf[i, j, :] = est_conf[i, j, :] / np.sum(est_conf[i, j, :])

    # Estimate marginal distribution q
    est_q = np.sum(pred_train, axis=0)
    est_q /= np.sum(est_q)

    # Estimate posterior distribution P, modified to incorporate copy rates
    for i in range(n):
        for j in workers_on_example[i]:
            if j in range(m): # added to filter out copycat
                if j == id_busy: # Busy user
                    est_label_posterior[i] += np.log(est_conf[j, :, :] @ response[i, j])
                else: # Copy cat
                    if np.all(response[i, j] == response[i, id_busy]): # Same as the busy user
                        weighted = est_conf[j, :, :] @ response[i, j] * (1 - est_copy_rates[j])
                        weighted += uniform_v * est_copy_rates[j]
                        # weighted += est_conf[id_busy, :, :] @ response[i, id_busy] * est_copy_rates[j]
                        est_label_posterior[i] += np.log(weighted)
                    else: # Different from the busy user
                        est_label_posterior[i] += np.log(est_conf[j, :, :] @ response[i, j])
        est_label_posterior[i] = np.exp(est_label_posterior[i]) * est_q
        est_label_posterior[i] /= np.sum(est_label_posterior[i])

    return est_q, est_label_posterior, est_conf, est_copy_rates
    
    
    