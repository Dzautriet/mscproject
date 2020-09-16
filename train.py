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
from MBEM import posterior_distribution
import models
import gc
import argparse
    
def call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, conf, copy_rates, two_stage=True, 
               use_pretrained=False, model=None, conf_init=None, use_aug=False, est_cr=True, reweight=True, dataset="mnist"):
    batch_size = 128
    epochs = 100
    _, m, k = labels_train.shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience = 15
    learning_rate = 0.001
    scale = 0.01
    save_path = 'model'
    save_path_cm = 'cm_layer'
    best_loss = np.inf
    best_epoch = 0
    verbose = 5
    early_stopped = False
    
    if reweight:
        redundancy = labels_train[0].sum()
        factor = (redundancy-1) / (m-1)
    else:
        factor = None
    
    # Data augmentation for CIFAR-10
    transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),       
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transforms_test_vali = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Create data loader
    X_train_tensor = torch.tensor(X_train[valid_range], dtype=torch.float)
    labels_tensor = torch.tensor(labels_train, dtype=torch.float)
    X_vali_tensor = torch.tensor(X_vali, dtype=torch.float)
    labels_vali_tensor = torch.tensor(labels_vali, dtype=torch.float)
    y_vali_tensor = torch.tensor(y_vali, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float)
    if use_aug:
        trainset = MyDataset(tensors=(X_train_tensor, labels_tensor), transforms=transforms_train)
        vali_corruptset = MyDataset(tensors=(X_vali_tensor, labels_vali_tensor), transforms=transforms_test_vali)
        valiset = MyDataset(tensors=(X_vali_tensor, y_vali_tensor), transforms=transforms_test_vali)
        testset = MyDataset(tensors=(X_test_tensor, y_test_tensor), transforms=transforms_test_vali)
    else:
        trainset = TensorDataset(X_train_tensor, labels_tensor)
        vali_corruptset = TensorDataset(X_vali_tensor, labels_vali_tensor)
        valiset = TensorDataset(X_vali_tensor, y_vali_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valiloader = DataLoader(valiset, batch_size=batch_size, shuffle=False, pin_memory=True)
    vali_corruptloader = DataLoader(vali_corruptset, batch_size=batch_size, shuffle=False, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    del X_train, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, X_train_tensor, labels_tensor, X_vali_tensor, labels_vali_tensor, y_vali_tensor, X_test_tensor, y_test_tensor
    
    if not use_pretrained:
        if dataset ==  "mnist":
            model = models.CNN_MNIST(k)
        elif dataset == "cifar10":
            model = models.CNN_CIFAR(torchvision.models.resnet18(pretrained=True, progress=True), k)
        else:
            pass
        model.to(device)
        confusion_matrices_layer = modelsConfMatLayer(m, k, est_cr, reweight, factor)
    else:
        confusion_matrices_layer = models.ConfMatLayer(m, k, est_cr, reweight, factor, conf_init)
    confusion_matrices_layer.to(device)
    # confusion_matrices_layer.calc_cm()
    # print(confusion_matrices_layer.confusion_matrices.detach().cpu().numpy())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer_cm = torch.optim.Adam(confusion_matrices_layer.parameters(), lr=learning_rate*20)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 500, 1000], gamma=0.1)
    
    for epoch in range(epochs):
        model.train()
        confusion_matrices_layer.train()
        train_loss = AverageMeter()
        vali_loss = AverageMeter()
        for batch_index, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            #%% Ordinary grad update
            if not two_stage:
                optimizer.zero_grad()
                optimizer_cm.zero_grad()
                log_softmax = model(inputs)
                weighted_xe = confusion_matrices_layer.forward(labels, log_softmax)
                # trace_norm = confusion_matrices_layer.trace_norm()
                # trace_norm = trace_norm.to(device)  
                
                # entropy = confusion_matrices_layer.entropy()
                # entropy = entropy.to(device)  
                # pred_entropy = (-torch.exp(log_softmax) * log_softmax).sum(axis=1).mean()  
                
                # total_loss = weighted_xe + scale * trace_norm
                # total_loss = weighted_xe + 0.1 * entropy
                total_loss = weighted_xe
                total_loss.backward()
                optimizer.step()
                optimizer_cm.step()
            #%% Two-stage grad update
            else:
                # Stage 1, update classifier
                optimizer.zero_grad()
                log_softmax = model(inputs)
                weighted_xe = confusion_matrices_layer.forward(labels, log_softmax)
                # trace_norm = confusion_matrices_layer.trace_norm()
                # trace_norm = trace_norm.to(device)
                
                # entropy = confusion_matrices_layer.entropy()
                # entropy = entropy.to(device)                
                
                # total_loss = weighted_xe + 0.1 * entropy
                # total_loss = weighted_xe + scale * trace_norm
                total_loss = weighted_xe
                total_loss.backward()
                optimizer.step()
                
                # Stage 2, update cm
                confusion_matrices_layer.reweight = False
                optimizer_cm.zero_grad()
                log_softmax = model(inputs)
                weighted_xe = confusion_matrices_layer.forward(labels, log_softmax)
                # trace_norm = confusion_matrices_layer.trace_norm()
                # trace_norm = trace_norm.to(device)
                
                # entropy of the output layer
                # pred_entropy = (-torch.exp(log_softmax) * log_softmax).sum(axis=1).mean()            
                
                total_loss = weighted_xe
                total_loss.backward()
                optimizer_cm.step()
            #%%
            # lr_scheduler.step()
            train_loss.update(total_loss.item(), inputs.size(0))
        # Evaluation on noisy vali
        model.eval()
        confusion_matrices_layer.eval()
        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(vali_corruptloader):
                inputs, labels = inputs.to(device), labels.to(device)
                log_softmax = model(inputs)
                weighted_xe = confusion_matrices_layer.forward(labels, log_softmax)
                trace_norm = confusion_matrices_layer.trace_norm()
                trace_norm = trace_norm.to(device)
                total_loss = weighted_xe + scale * trace_norm
                vali_loss.update(total_loss.item(), inputs.size(0))
                    
        if epoch % verbose == 0:
            print("Epoch: {}/{}, training loss: {:.4f}, vali loss: {:.4f}.".format(epoch, epochs, train_loss.avg, vali_loss.avg))
            # print(pred_entropy)
        
        # Saving best
        if vali_loss.avg < best_loss:
            best_loss = vali_loss.avg
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            torch.save(confusion_matrices_layer.state_dict(), save_path_cm)
        # Early stopping
        if epoch - best_epoch >= patience:
            print("Early stopping at epoch {}!".format(epoch))
            model.load_state_dict(torch.load(save_path))
            confusion_matrices_layer.load_state_dict(torch.load(save_path_cm))
            early_stopped = True
            print("Restoring models from epoch {}...".format(best_epoch))
        
        if epoch % 10 == 0 or early_stopped or epoch >= epochs - 1:
            # Evaluation on GT
            model.eval()
            confusion_matrices_layer.eval()
            vali_acc = AverageMeter()
            test_acc = AverageMeter()
            with torch.no_grad():
                for batch_index, (inputs, targets) in enumerate(valiloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    acc = accuracy(outputs, targets)
                    vali_acc.update(acc, inputs.size(0))
                for batch_index, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    acc = accuracy(outputs, targets)
                    test_acc.update(acc, inputs.size(0))
                confusion_matrices_layer.calc_cm() # update confusion matrices
            # Calculate CM diff
            est_conf = confusion_matrices_layer.confusion_matrices.detach().cpu().numpy()
            est_copyrates = confusion_matrices_layer.copyrates.detach().cpu().numpy()
            mean_cm_diff = np.mean([np.linalg.norm(est_conf[i] - conf[i]) for i in range(m)])
            mean_cp_diff = np.mean(np.abs(est_copyrates[1:] - copy_rates[1:]))
            print("Vali accuracy :{:.4f}, test accuracy: {:.4f}, mean CM est error: {:.4f}, mean CP est error: {:.4f}."
                  .format(vali_acc.avg, test_acc.avg, mean_cm_diff, mean_cp_diff))
            
            # plot_conf_mat(est_conf, conf)
            
        if early_stopped:
            break
        
    return est_conf, est_copyrates, test_acc.avg, mean_cm_diff, mean_cp_diff


