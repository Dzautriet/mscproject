# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:37:36 2020

@author: Zhe Cao
"""

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
import models
import gc
%matplotlib inline

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

#%% Functions & classes
class MyDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transforms=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transforms:
            x = self.transforms(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class ConfMatLayer(nn.Module):
    def __init__(self, m, k):
        super(ConfMatLayer, self).__init__()
        w_init = torch.tensor(np.stack([6.*np.eye(k)-5. for i in range(m)]), dtype=torch.float32)
        self.p = nn.Parameter(w_init, requires_grad=True)
    
    def calc_cm(self):
        rho = F.softplus(self.p)
        self.confusion_matrices = rho / rho.sum(axis=-1, keepdims=True)
        # self.confusion_matrices = F.softmax(self.p, dim=-1)
    
    def trace_norm(self):
        traces = torch.tensor([torch.trace(cm) for cm in torch.unbind(self.confusion_matrices, axis=0)], requires_grad=True)
        return traces.mean()
    
    def forward(self, labels, logsoftmax):
        """
        labels: n * m * k
        logsoftmax: n * k
        """
        self.calc_cm()            
        losses_all_users = []
        for idx, labels_i in enumerate(torch.unbind(labels, axis=1)):
            preds_true = torch.exp(logsoftmax) # n * k
            preds_user = torch.matmul(preds_true, self.confusion_matrices[idx, :, :])
            preds_clipped = torch.clamp(preds_user, 1e-10, 0.9999999)
            loss = -labels_i * torch.log(preds_clipped) # n * k
            loss = loss.sum(axis=1) # n
            losses_all_users.append(loss)
        losses_all_users = torch.stack(losses_all_users, axis=1) # n * m
        has_labels = torch.sum(labels, axis=2) # n * m
        losses_all_users *= has_labels
        losses_all_users = torch.mean(torch.sum(losses_all_users, axis=1))
        return losses_all_users
        

#%% Filter out copycat's response
m = 5
gamma = .35
repeat = 2
valid_range = np.arange(50000)

conf = generate_conf_pairflipper(m, k, gamma)

# labels_train, _, workers_on_example = generate_labels_weight(y_train[valid_range], repeat, conf)
# labels_vali, _, workers_on_example_vali = generate_labels_weight(y_vali, repeat, conf)

copy_rates = np.zeros(m)
num_busy = 1 # 1 by default, starting from no.0
labels_train, _, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
labels_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)

#%% Training function
use_pretrained = False
model = None
use_aug = False
# def call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, use_pretrained=False, model=None, use_aug=False):
batch_size = 128
epochs = 1000
_, m, k = labels_train.shape
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
patience = 30
learning_rate = 0.01
scale = 0.01
save_path = 'model'
save_path_cm = 'cm_layer'
best_loss = np.inf
best_epoch = 0
verbose = 1
early_stopped = False

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
labels_tensor = torch.tensor(labels_train, dtype=torch.float)
X_vali_tensor = torch.tensor(X_vali, dtype=torch.float)
labels_vali_tensor = torch.tensor(labels_vali, dtype=torch.float)
y_vali_tensor = torch.tensor(y_vali, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)
if use_aug:
    trainset = MyDataset(tensors=(X_train_tensor, labels_tensor), transforms=transforms_train)
    trainset_pred = MyDataset(tensors=(X_train_tensor_pred,), transforms=transforms_test_vali)
    vali_corruptset = MyDataset(tensors=(X_vali_tensor, labels_vali_tensor), transforms=transforms_test_vali)
    valiset = MyDataset(tensors=(X_vali_tensor, y_vali_tensor), transforms=transforms_test_vali)
    testset = MyDataset(tensors=(X_test_tensor, y_test_tensor), transforms=transforms_test_vali)
else:
    trainset = TensorDataset(X_train_tensor, labels_tensor)
    trainset_pred = TensorDataset(X_train_tensor_pred,)
    vali_corruptset = TensorDataset(X_vali_tensor, labels_vali_tensor)
    valiset = TensorDataset(X_vali_tensor, y_vali_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
trainloader_pred = DataLoader(trainset_pred, batch_size=batch_size, shuffle=False, pin_memory=True)
valiloader = DataLoader(valiset, batch_size=batch_size, shuffle=False, pin_memory=True)
vali_corruptloader = DataLoader(vali_corruptset, batch_size=batch_size, shuffle=False, pin_memory=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
# del X_train, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, X_train_tensor, labels_tensor, X_vali_tensor, labels_vali_tensor, y_vali_tensor, X_test_tensor, y_test_tensor

if not use_pretrained:
    # model = resnet_pytorch.resnet20()
    # model = resnet_pytorch_2.ResNet18()
    model = models.CNN_MNIST()
    model.to(device)
    
confusion_matrices_layer = ConfMatLayer(m, k)
confusion_matrices_layer.to(device)

# optimizer = torch.optim.Adam(list(model.parameters())+list(confusion_matrices_layer.parameters()), lr=learning_rate, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
optimizer_cm = torch.optim.Adam(confusion_matrices_layer.parameters(), lr=learning_rate*20)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 500, 1000], gamma=0.1)

for epoch in range(epochs):
    model.train()
    confusion_matrices_layer.train()
    train_loss = AverageMeter()
    vali_loss = AverageMeter()
    for batch_index, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        optimizer_cm.zero_grad()
        log_softmax = model(inputs)
        weighted_xe = confusion_matrices_layer.forward(labels, log_softmax)
        trace_norm = confusion_matrices_layer.trace_norm()
        trace_norm = trace_norm.to(device)
        total_loss = weighted_xe + scale * trace_norm
        total_loss.backward()
        optimizer.step()
        optimizer_cm.step()
        lr_scheduler.step()
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
        mean_diff = np.mean([np.linalg.norm(est_conf[i] - conf[i]) for i in range(m)])
        print("Vali accuracy :{:.4f}, test accuracy: {:.4f}, mean CM est error: {:.4f}."
              .format(vali_acc.avg, test_acc.avg, mean_diff))
    
    if early_stopped:
        break
        
#     return est_conf
        
# est_conf = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, use_pretrained=False, model=None, use_aug=False)
 
plot_conf_mat(est_conf, conf)

