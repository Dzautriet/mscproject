# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:44:36 2020

@author: Zhe Cao

Copycat scenario: systematic comparison
Reweighting mechanism
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
from train import call_train as call_train_mbem
from MBEM import posterior_distribution
import models
import gc
# %matplotlib inline

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


def call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, conf, copy_rates, use_pretrained=False, model=None, use_aug=False, est_cr=True, reweight=True):
    batch_size = 128
    epochs = 100
    _, m, k = labels_train.shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience = 15
    learning_rate = 0.01
    scale = 0.01
    save_path = 'model'
    save_path_cm = 'cm_layer'
    best_loss = np.inf
    best_epoch = 0
    verbose = 5
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
        # model = models.CNN_CIFAR(torchvision.models.resnet18(pretrained=True))
        model.to(device)
    
    if reweight:
        redundancy = labels_train[0].sum()
        factor = (redundancy-1) / (m-1)
    else:
        factor = None
    confusion_matrices_layer = models.ConfMatLayer(m, k, est_cr, reweight, factor)
    confusion_matrices_layer.to(device)
    
    # optimizer = torch.optim.Adam(list(model.parameters())+list(confusion_matrices_layer.parameters()), lr=learning_rate, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    optimizer_cm = torch.optim.Adam(confusion_matrices_layer.parameters(), lr=learning_rate*20)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 500, 1000], gamma=0.1)
    
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
        
        if early_stopped:
            break
        
    return est_conf, est_copyrates, test_acc.avg, mean_cm_diff, mean_cp_diff
        
def plot_result_std_cr(arrays, copy_rate_range, title, ylabel, filename):
    """
    arrays: num_rep * num_copyrates * 4
    standard deviation
    """
    avg = arrays.mean(axis=0)
    std = arrays.std(axis=0)
    lower = std
    upper= std
    plt.errorbar(x=copy_rate_range, y=avg[:, 0], yerr=[lower[:, 0], upper[:, 0]], label="w/o copy rate est.", fmt='-o')
    plt.errorbar(x=copy_rate_range, y=avg[:, 1], yerr=[lower[:, 1], upper[:, 1]], label="w/ copy rate est.", fmt='-o')
    plt.errorbar(x=copy_rate_range, y=avg[:, 2], yerr=[lower[:, 2], upper[:, 2]], label="copy rate est & reweight.", fmt='-o')
    plt.errorbar(x=copy_rate_range, y=avg[:, 3], yerr=[lower[:, 3], upper[:, 3]], label="majority vote.", fmt='-o')
    plt.errorbar(x=copy_rate_range, y=avg[:, 4], yerr=[lower[:, 4], upper[:, 4]], label="MBEM.", fmt='-o')
    plt.ylim(0.0, arrays.max()+0.01)
    plt.title(title)
    plt.xlabel("Copy probability")
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(filename+".png", bbox_inches='tight')
    np.save(filename+".npy", arrays)
    plt.show()

#%% Main
if __name__ == "__main__":
    m = 5 # number of users
    gamma_b = .30 # skill level of the busy user
    gamma_c = .30 # skill level of the other users
    repeat = 2 # redundancy
    valid_range = np.arange(50000)
    num_busy = 1 # 1 by default, starting from No.0
    copy_rates = np.zeros(m)
    copy_ids = np.arange(1, 2) # user no.1 is the copycat
    copy_rate_range = np.arange(0.1, 1.0, 0.2)
    num_rep = 5 # repetition
    test_accs = np.zeros((num_rep, len(copy_rate_range), 5)) # five algorithms to compare
    conf_errors = np.zeros((num_rep, len(copy_rate_range), 5))
    cp_errors = np.zeros((num_rep, len(copy_rate_range), 5))
    
    title = "Redundancy:{}, skill level: {}".format(repeat, gamma_c)
    filename = "Copyrate_layer_lossreweight_r_{}_g_{}_5comp".format(repeat, gamma_c).replace('.', '')
    
    for rep in range(num_rep):
        print("Repetition: {}".format(rep))
        for i, copy_rate_value in enumerate(copy_rate_range):
            copy_rates[copy_ids] = copy_rate_value
            print("----------------")
            print("Copy rates: {}".format(copy_rates))
            conf_b = generate_conf_pairflipper(1, k, gamma_b)
            conf = generate_conf_pairflipper(m, k, gamma_c)
            conf[:1] = conf_b
            labels_train, _, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
            labels_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)
            
            # 1. Estimating copy rates & reweighting
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, use_pretrained=False, model=None, use_aug=False, est_cr=True, reweight=True)
            test_accs[rep, i, 2] = test_acc
            conf_errors[rep, i, 2] = conf_error
            cp_errors[rep, i, 2] = cp_error
            print(est_copyrates[1:])
            plot_conf_mat(est_conf, conf)
            
            print("--------")
            # 2. Estimating copy rates
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, use_pretrained=False, model=None, use_aug=False, est_cr=True, reweight=False)
            test_accs[rep, i, 1] = test_acc
            conf_errors[rep, i, 1] = conf_error
            cp_errors[rep, i, 1] = cp_error
            
            print("--------")
            # 3. not estimating copy rates
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, use_pretrained=False, model=None, use_aug=False, est_cr=False, reweight=False)
            test_accs[rep, i, 0] = test_acc
            conf_errors[rep, i, 0] = conf_error
            cp_errors[rep, i, 0] = cp_error
            
            print("--------")
            # 4. & 5. MBEM
            y_train_wmv = np.sum(labels_train, axis=1) / repeat
            y_vali_corrupt = np.sum(labels_vali, axis=1) / repeat
            pred_train, pred_vali, vali_acc, test_acc, model = call_train_mbem(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug)
            test_accs[rep, i, 3] = test_acc
            conf_errors[rep, i, 3] = conf_error
            cp_errors[rep, i, 3] = cp_error
            for j in range(1):
                est_q, est_label_posterior, est_conf = posterior_distribution(labels_train, pred_train, workers_on_example)
                est_q_vali, est_label_posterior_vali, _ = posterior_distribution(labels_vali, pred_vali, workers_on_example_vali)
                # Train
                pred_train, pred_vali, vali_acc, test_acc, model = call_train_mbem(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug)
                test_accs[rep, i, 4] = test_acc
                conf_errors[rep, i, 4] = conf_error
            cp_errors[rep, i, 4] = cp_error
            
    plot_result_std_cr(test_accs, copy_rate_range, title=title, ylabel="Test accuracy", filename=filename+"_testacc")
    plot_result_std_cr(cp_errors, copy_rate_range, title=title, ylabel="Copy probability estimation error", filename=filename+"_cperror")
    plot_result_std_cr(conf_errors, copy_rate_range, title=title, ylabel="Confusion matrix estimation error", filename=filename+"_conferror")
