# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:14:00 2020

@author: Zhe Cao
"""

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_pytorch
import gc

#%% Load data
X = np.load("./cifar10/X.npy")
y = np.load("./cifar10/y.npy")
X_test = np.load("./cifar10/X_test.npy")
y_test = np.load("./cifar10/y_test.npy")

#%% Convert data
k = 10
X = X / 255.0
X_test = X_test / 255.0
y = y.astype(int)
y_test = y_test.astype(int)
y = np.eye(k)[y]
y_test = np.eye(k)[y_test]
X_train, X_vali = X[:45000], X[45000:]
y_train, y_vali = y[:45000], y[45000:]

#%% Create data loader
batch_size = 64
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_vali_tensor = torch.tensor(X_vali, dtype=torch.float)
y_vali_tensor = torch.tensor(y_vali, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
trainset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
valiset = torch.utils.data.TensorDataset(X_vali_tensor, y_vali_tensor)
testset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
valiloader = torch.utils.data.DataLoader(valiset, batch_size=batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
del X, y, X_train, y_train, X_vali, y_vali, X_test, y_test, X_train_tensor, y_train_tensor, X_vali_tensor, y_vali_tensor, X_test_tensor, y_test_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Modified crossentropy loss function for soft labels
def XE(predicted, target):
    return -(target * predicted).sum(dim=1).mean()

#%% Accuracy
def accuracy(pred, targets):
    num_correct = pred.argmax(dim=1).eq(targets.argmax(dim=1)).sum()
    acc = num_correct.float() / targets.size(0)
    return acc

#%% AverageMeter class
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
#%% Traning function
# model = torchvision.models.resnet18()
epochs = 200
model = resnet_pytorch.resnet20()
model.to(device)
save_path = 'model'
best_acc = 0
best_epoch = 0
patience = 20

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

for epoch in range(3):
    model.train()
    train_loss = AverageMeter()
    vali_loss = AverageMeter()
    vali_acc = AverageMeter()
    test_acc = AverageMeter()
    for batch_index, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = XE(outputs, targets)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss.update(loss.item(), inputs.size(0))            
        
    # Evaluation
    model.eval()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(valiloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = XE(outputs, targets)
            vali_loss.update(loss.item(), inputs.size(0))
            acc = accuracy(outputs, targets)
            vali_acc.update(acc, inputs.size(0))
        
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc = accuracy(outputs, targets)
            test_acc.update(acc, inputs.size(0))        
    print("Epoch: {}/{}, training loss: {:.4f}, vali loss: {:.4f}, vali acc: {:.4f}, test acc: {:.4f}.".format(epoch, epochs, train_loss.avg, vali_loss.avg, vali_acc.avg, test_acc.avg))
    
    # Saving best
    if vali_acc.avg > best_acc:
        best_acc = vali_acc.avg
        best_epoch = epoch
        torch.save(model.state_dict(), save_path)
        
    # Early stopping
    if epoch - best_epoch >= patience:
        print("Early stopping")
        break
    
# Resume best model and output prediction
model.load_state_dict(torch.load(save_path))
model.eval()
pred_train = torch.empty(size=(0, targets.size(1))).to(device)
pred_vali = torch.empty(size=(0, targets.size(1))).to(device)

with torch.no_grad():
    for batch_index, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        pred_train = torch.cat((pred_train, outputs), dim=0)
        
    for batch_index, (inputs, targets) in enumerate(valiloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        pred_vali = torch.cat((pred_vali, outputs), dim=0)
        
pred_train = pred_train.cpu().numpy()
pred_vali = pred_vali.cpu().numpy()
    
    
    
    
    
    
    
    
    
    