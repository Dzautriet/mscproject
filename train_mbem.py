# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:02:43 2020

@author: Zhe Cao

Reimplemention of MBEM
(Khetan, A., Lipton, Z. C. & Anandkumar, A. (2018), 
Learning from noisy singly-labeled data, 
in ‘International Conference on Learning Representations’.)
(https://github.com/khetan2/MBEM)
"""

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader
import models
import gc
from utils import AverageMeter, MyDataset
        

def XE(pred, target):
    """
    pred: log softmax output
    target: labels
    """
    return -(target * pred).sum(dim=1).mean()

def accuracy(pred, targets):
    """
    pred: log softmax output
    target: labels
    """
    num_correct = pred.argmax(dim=1).eq(targets.argmax(dim=1)).sum()
    acc = num_correct.float() / targets.size(0)
    return acc

def call_train(X_train, valid_range, y_train_corrupt, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_pretrained=False, model=None, use_aug=False, dataset="mnist"):
    batch_size = 128
    epochs = 100
    k = y_train_corrupt.shape[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience = 15
    if dataset == "mnist":
        learning_rate = 0.01
    elif dataset == "cifar10":
        learning_rate = 0.001 # 0.01 for MNIST, 0.001 for cifar-10
    else:
        pass
    save_path = 'model'
    best_acc = 0
    best_epoch = 0
    num_classes = y_train_corrupt.shape[1]
    verbose = 10
    
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
    X_train_tensor_pred = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train_corrupt, dtype=torch.float)
    X_vali_tensor = torch.tensor(X_vali, dtype=torch.float)
    y_vali_corrupt_tensor = torch.tensor(y_vali_corrupt, dtype=torch.float)
    y_vali_tensor = torch.tensor(y_vali, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float)
    if use_aug:
        trainset = MyDataset(tensors=(X_train_tensor, y_train_tensor), transforms=transforms_train)
        trainset_pred = MyDataset(tensors=(X_train_tensor_pred, ), transforms=transforms_test_vali)
        vali_corruptset = MyDataset(tensors=(X_vali_tensor, y_vali_corrupt_tensor), transforms=transforms_test_vali)
        valiset = MyDataset(tensors=(X_vali_tensor, y_vali_tensor), transforms=transforms_test_vali)
        testset = MyDataset(tensors=(X_test_tensor, y_test_tensor), transforms=transforms_test_vali)
    else:
        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainset_pred = TensorDataset(X_train_tensor_pred, )
        vali_corruptset = TensorDataset(X_vali_tensor, y_vali_corrupt_tensor)
        valiset = TensorDataset(X_vali_tensor, y_vali_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    trainloader_pred = DataLoader(trainset_pred, batch_size=batch_size, shuffle=False, pin_memory=True)
    valiloader = DataLoader(valiset, batch_size=batch_size, shuffle=False, pin_memory=True)
    vali_corruptloader = DataLoader(vali_corruptset, batch_size=batch_size, shuffle=False, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    del X_train, y_train_corrupt, X_vali, y_vali_corrupt, y_vali, X_test, y_test, X_train_tensor, y_train_tensor, X_vali_tensor, y_vali_tensor, y_vali_corrupt_tensor, X_test_tensor, y_test_tensor
    
    if not use_pretrained:
        # model = resnet_pytorch.resnet20()
        # model = resnet_pytorch_2.ResNet18()
        if dataset ==  "mnist":
            model = models.CNN_MNIST(k)
        elif dataset == "cifar10":
            model = models.CNN_CIFAR(torchvision.models.resnet18(pretrained=True, progress=True), k)
        else:
            pass
        model.to(device)
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 120])
                                                                                                                                                                                                                                                                          
    for epoch in range(epochs):
        model.train()
        train_loss = AverageMeter()
        vali_loss = AverageMeter()
        vali_acc = AverageMeter()
        for batch_index, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = XE(outputs, targets)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            train_loss.update(loss.item(), inputs.size(0))            
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(vali_corruptloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = XE(outputs, targets)
                vali_loss.update(loss.item(), inputs.size(0))
                acc = accuracy(outputs, targets)
                vali_acc.update(acc, inputs.size(0))
        if epoch % verbose == 0:
            print("Epoch: {}/{}, training loss: {:.4f}, vali loss: {:.4f}, vali acc: {:.4f}.".format(epoch, epochs, train_loss.avg, vali_loss.avg, vali_acc.avg))
        
        # Saving best
        if vali_acc.avg > best_acc:
            best_acc = vali_acc.avg
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
        # Early stopping
        if epoch - best_epoch >= patience:
            print("Early stopping at epoch {}!".format(epoch))
            break
        
    # Resume best model and output prediction
    model.load_state_dict(torch.load(save_path))
    model.eval()
    pred_train = torch.empty(size=(0, targets.size(1))).to(device)
    pred_vali = torch.empty(size=(0, targets.size(1))).to(device)
    vali_acc = AverageMeter()
    test_acc = AverageMeter()
    with torch.no_grad():
        for batch_index, (inputs,) in enumerate(trainloader_pred):
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_train = torch.cat((pred_train, outputs), dim=0)
        for batch_index, (inputs, targets) in enumerate(valiloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc = accuracy(outputs, targets)
            vali_acc.update(acc, inputs.size(0))
            pred_vali = torch.cat((pred_vali, outputs), dim=0)
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc = accuracy(outputs, targets)
            test_acc.update(acc, inputs.size(0))
                
    # pred_train = torch.exp(pred_train)
    # pred_vali = torch.exp(pred_vali)
    pred_train = pred_train.cpu().numpy()
    pred_vali = pred_vali.cpu().numpy()
    pred_train = np.eye(num_classes)[np.argmax(pred_train, axis=1)]
    pred_vali = np.eye(num_classes)[np.argmax(pred_vali, axis=1)]
    
    print("True vali accuracy :{:.4f}, test accuracy: {:.4f}.".format(vali_acc.avg, test_acc.avg))
    
    return pred_train, pred_vali, vali_acc.avg.cpu().numpy(), test_acc.avg.cpu().numpy(), model     

def call_train_blobs(X_train, valid_range, y_train_corrupt, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_pretrained=False, model=None):
    batch_size = 128
    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience = 5
    learning_rate = 0.01
    save_path = 'model_blobs'
    best_acc = 0
    best_epoch = 0
    n_features = X_train.shape[1]
    num_classes = y_train_corrupt.shape[1]
    verbose = 1
    
    # Create data loader
    X_train_tensor = torch.tensor(X_train[valid_range], dtype=torch.float)
    X_train_tensor_pred = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train_corrupt, dtype=torch.float)
    X_vali_tensor = torch.tensor(X_vali, dtype=torch.float)
    y_vali_corrupt_tensor = torch.tensor(y_vali_corrupt, dtype=torch.float)
    y_vali_tensor = torch.tensor(y_vali, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float)

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    trainset_pred = TensorDataset(X_train_tensor_pred,)
    vali_corruptset = TensorDataset(X_vali_tensor, y_vali_corrupt_tensor)
    valiset = TensorDataset(X_vali_tensor, y_vali_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    trainloader_pred = DataLoader(trainset_pred, batch_size=batch_size, shuffle=False, pin_memory=True)
    valiloader = DataLoader(valiset, batch_size=batch_size, shuffle=False, pin_memory=True)
    vali_corruptloader = DataLoader(vali_corruptset, batch_size=batch_size, shuffle=False, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    del X_train, y_train_corrupt, X_vali, y_vali_corrupt, y_vali, X_test, y_test, X_train_tensor, y_train_tensor, X_vali_tensor, y_vali_tensor, y_vali_corrupt_tensor, X_test_tensor, y_test_tensor
    
    if not use_pretrained:
        model = models.MLP_BLOBS(n_features, num_classes)
        model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
                                                                                                                                                                                                                                                                          
    for epoch in range(epochs):
        model.train()
        train_loss = AverageMeter()
        vali_loss = AverageMeter()
        vali_acc = AverageMeter()
        for batch_index, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = XE(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), inputs.size(0))            
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(vali_corruptloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = XE(outputs, targets)
                vali_loss.update(loss.item(), inputs.size(0))
                acc = accuracy(outputs, targets)
                vali_acc.update(acc, inputs.size(0))
        if epoch % verbose == 0:
            print("Epoch: {}/{}, training loss: {:.4f}, vali loss: {:.4f}, vali acc: {:.4f}.".format(epoch, epochs, train_loss.avg, vali_loss.avg, vali_acc.avg))
        
        # Saving best
        if vali_acc.avg > best_acc:
            best_acc = vali_acc.avg
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
        # Early stopping
        if epoch - best_epoch >= patience:
            print("Early stopping at epoch {}!".format(epoch))
            break
        
    # Resume best model and output prediction
    model.load_state_dict(torch.load(save_path))
    model.eval()
    pred_train = torch.empty(size=(0, targets.size(1))).to(device)
    pred_vali = torch.empty(size=(0, targets.size(1))).to(device)
    vali_acc = AverageMeter()
    test_acc = AverageMeter()
    with torch.no_grad():
        for batch_index, (inputs,) in enumerate(trainloader_pred):
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_train = torch.cat((pred_train, outputs), dim=0)
        for batch_index, (inputs, targets) in enumerate(valiloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc = accuracy(outputs, targets)
            vali_acc.update(acc, inputs.size(0))
            pred_vali = torch.cat((pred_vali, outputs), dim=0)
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc = accuracy(outputs, targets)
            test_acc.update(acc, inputs.size(0))
                
    # pred_train = torch.exp(pred_train)
    # pred_vali = torch.exp(pred_vali)
    pred_train = pred_train.cpu().numpy()
    pred_vali = pred_vali.cpu().numpy()
    pred_train = np.eye(num_classes)[np.argmax(pred_train, axis=1)]
    pred_vali = np.eye(num_classes)[np.argmax(pred_vali, axis=1)]
    
    print("True vali accuracy :{:.4f}, test accuracy: {:.4f}.".format(vali_acc.avg, test_acc.avg))
    
    return pred_train, pred_vali, vali_acc.avg.cpu().numpy(), test_acc.avg.cpu().numpy(), model     

