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
from train import call_train as call_train_mbem
from MBEM import posterior_distribution
import models
import gc
import argparse

#%% Functions & classes
class ConfMatLayer(nn.Module):
    """
    Adapted for ablation study
    """
    def __init__(self, m, k, est_cr=True, reweight=False, factor=None, conf_init=None):
        super(ConfMatLayer, self).__init__()
        self.m = m
        self.est_cr = est_cr # Whether estimate copy rates or not
        self.reweight = reweight # Whether reweight losses according to copy rates and label counts or not
        self.factor = factor # reweighting factor that accounts for sparsity
        if conf_init is None:
            b_init = torch.tensor(np.stack([6.*np.eye(k)-5. for i in range(m)]), dtype=torch.float32)
            # b_init = torch.tensor(np.stack([2.43*np.eye(k)-2.86 for i in range(m)]), dtype=torch.float32)
            self.b = nn.Parameter(b_init, requires_grad=True)
        else:
            # Initialise from pretrained confmat
            # b_init = np.stack([6.*np.eye(k)-5. for i in range(m)])     
            # b_init[0] = np.log(np.exp(np.clip(conf_init[0], 1e-10, 1.))-1) # Inverse of softplus
            b_init = np.log(np.exp(np.clip(conf_init, 1e-10, 1.))-1) # Inverse of softplus
            b_init = torch.tensor(b_init, dtype=torch.float32)
            self.b = nn.Parameter(b_init, requires_grad=True)
        if self.est_cr:
            theta_init = torch.tensor(np.ones(m) * -5, dtype=torch.float32)
            self.theta = nn.Parameter(theta_init, requires_grad=True)
    
    def calc_cm(self):
        rho = F.softplus(self.b)
        self.confusion_matrices = rho / rho.sum(axis=-1, keepdims=True)
        if self.est_cr:
            self.copyrates = torch.sigmoid(self.theta)
        else:
            self.copyrates = torch.tensor(np.zeros(self.m), dtype=torch.float32) # if not, fix copy rates to zeros
    
    def trace_norm(self):
        traces = torch.tensor([torch.trace(cm) for cm in torch.unbind(self.confusion_matrices, axis=0)], requires_grad=True)
        return traces.mean()
    
    def entropy(self):
        '''Entroy of the estimated confusion matrices'''
        log_product = -self.confusion_matrices * torch.log(torch.clamp(self.confusion_matrices, 1e-10, 1))
        # entropy = torch.tensor(entropy.sum() / self.m, requires_grad=True)
        log_product = log_product.sum() / self.m
        entropy = torch.tensor(log_product, requires_grad=True)
        return entropy
        
    def forward(self, labels, logsoftmax):
        """
        labels: n * m * k
        logsoftmax: n * k
        Works with copy probs
        """
        self.calc_cm()            
        losses_all_users = []
        preds_true = torch.exp(logsoftmax) # n * k
        for idx, labels_i in enumerate(torch.unbind(labels, axis=1)):
            preds_user_intrinsic = torch.matmul(preds_true, self.confusion_matrices[idx, :, :])
            if idx == 0:
                # busy user
                preds_busy = preds_user_intrinsic
                preds_user = preds_user_intrinsic
            else:
                preds_user = preds_busy * self.copyrates[idx] + preds_user_intrinsic * (1 - self.copyrates[idx])
            preds_clipped = torch.clamp(preds_user, 1e-10, 0.9999999)
            loss = -labels_i * torch.log(preds_clipped) # n * k
            loss = loss.sum(axis=1) # n
            losses_all_users.append(loss)
        losses_all_users = torch.stack(losses_all_users, axis=1) # n * m
        has_labels = torch.sum(labels, axis=2) # n * m
        losses_all_users *= has_labels # n * m
        
        if self.reweight:
            with torch.no_grad():
                losses_weight = torch.ones(self.m).cuda()
                if self.reweight == "GAMMA+BOTH":
                    est_gamma = torch.tensor([torch.diag(self.confusion_matrices[i]).mean() for i in range(self.m)]).to(self.b.device)
                    factor_gamma = est_gamma[0] / (est_gamma[1:].mean())
                    if factor_gamma < 1.5:
                        # losses_weight[0] -= self.copyrates[1:].sum()
                        losses_weight[0] -= torch.clamp(self.copyrates[1:].sum(), 0, 1) # changed from sum to mean ,added clipping
                        losses_weight[0] *= self.factor
                    else:
                        losses_weight[0] *= factor_gamma
                elif self.reweight == "BOTH":
                    # losses_weight[0] -= self.copyrates[1:].sum()
                    losses_weight[0] -= torch.clamp(self.copyrates[1:].sum(), 0, 1)
                    losses_weight[0] *= self.factor
                elif self.reweight == "GAMMA+CNT":
                    est_gamma = torch.tensor([torch.diag(self.confusion_matrices[i]).mean() for i in range(self.m)]).to(self.b.device)
                    factor_gamma = est_gamma[0] / (est_gamma[1:].mean())
                    if factor_gamma < 1.5:
                        losses_weight[0] *= self.factor
                    else:
                        losses_weight[0] *= factor_gamma
                elif self.reweight == "GAMMA+CP":
                    est_gamma = torch.tensor([torch.diag(self.confusion_matrices[i]).mean() for i in range(self.m)]).to(self.b.device)
                    factor_gamma = est_gamma[0] / (est_gamma[1:].mean())
                    if factor_gamma < 1.5:
                        # losses_weight[0] -= self.copyrates[1:].sum()
                        losses_weight[0] -= torch.clamp(self.copyrates[1:].sum(), 0, 1)
                    else:
                        losses_weight[0] *= factor_gamma
                else:
                    raise ValueError("Illegal value for reweight!")
                losses_weight = torch.clamp(losses_weight, 1e-3)
            losses_all_users *= losses_weight
        
        losses_all_users = torch.mean(torch.sum(losses_all_users, axis=1))
        return losses_all_users

    
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
        confusion_matrices_layer = ConfMatLayer(m, k, est_cr, reweight, factor)
    else:
        confusion_matrices_layer = ConfMatLayer(m, k, est_cr, reweight, factor, conf_init)
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
        
    
def plot_result_std_cr(arrays, copy_rate_range, title, ylabel, filename):
    """
    arrays: num_rep * num_copyrates * 4
    standard deviation
    """
    avg = arrays.mean(axis=0)
    std = arrays.std(axis=0)
    lower = std
    upper= std
    plt.errorbar(x=copy_rate_range, y=avg[:, 0], yerr=[lower[:, 0], upper[:, 0]], capsize=4, label="proposed method", fmt='--o', color=u'#1f77b4')
    if np.any(avg[:, 1] != 0):
        plt.errorbar(x=copy_rate_range, y=avg[:, 1], yerr=[lower[:, 1], upper[:, 1]], capsize=4, label="vanilla", fmt='--o', color=u'#ff7f0e')
    if np.any(avg[:, 2] != 0):
        plt.errorbar(x=copy_rate_range, y=avg[:, 2], yerr=[lower[:, 2], upper[:, 2]], capsize=4, label="majority vote", fmt='--o', color=u'#2ca02c')
    if np.any(avg[:, 3] != 0):
        plt.errorbar(x=copy_rate_range, y=avg[:, 3], yerr=[lower[:, 3], upper[:, 3]], capsize=4, label="MBEM", fmt='--o', color=u'#d62728')
    plt.ylim(0.0, arrays.max()+0.01)
    plt.title(title)
    plt.xlabel("Copy probability")
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(filename+".png", bbox_inches='tight')
    np.save(filename+".npy", arrays)
    # plt.show()
    plt.close()

#%% Main
if __name__ == "__main__":
    #%% Parse arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('gamma_b', type=float, help='skill level of the busy user')
    parser.add_argument('gamma_c', type=float, help='skill level of the other users')
    parser.add_argument('dataset', type=str, help='mnist or cifar10')
    args = parser.parse_args()
    
    dataset = args.dataset
    #%% Load cifar-10 data
    if dataset == 'cifar10':    
        X = np.load("./cifar10/X.npy")
        y = np.load("./cifar10/y.npy")
        X_test = np.load("./cifar10/X_test.npy")
        y_test = np.load("./cifar10/y_test.npy")        
        k = 10
        X = X / 255.0
        X_test = X_test / 255.0
        y = y.astype(int)
        y_test = y_test.astype(int)
        y = np.eye(k)[y]
        y_test = np.eye(k)[y_test]
        X_train, X_vali = X[:45000], X[45000:]
        y_train, y_vali = y[:45000], y[45000:]
        use_aug = False    
    #%% Load MNIST data
    elif dataset == 'mnist':
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
    else:
        raise ValueError("Invalid value for dataset!")

    m = 5 # number of users
    # gamma_b = .35 # skill level of the busy user
    # gamma_c = .35 # skill level of the other users
    gamma_b = args.gamma_b
    gamma_c = args.gamma_c
    repeat = 2 # redundancy
    if dataset == 'cifar10':
        valid_range = np.arange(45000)
    elif dataset == 'mnist':
        valid_range = np.arange(50000) # max. 50000
    else:
        pass
    print("Training on {} samples.".format(len(valid_range)))
    num_busy = 1 # 1 by default, starting from No.0
    copy_rates = np.zeros(m)
    copy_ids = np.arange(1, 2) # user no.1 is the copycat
    copy_rate_range = np.arange(0.0, 1.01, 0.25)
    num_rep = 3 # repetition
    mbem_round = 1
    test_accs = np.zeros((num_rep, len(copy_rate_range), 4)) # four algorithms to compare
    conf_errors = np.zeros((num_rep, len(copy_rate_range), 4))
    cp_errors = np.zeros((num_rep, len(copy_rate_range), 4))
    
    title = "Redundancy:{}, skill level: {} & {}".format(repeat, gamma_b, gamma_c)
    filename = "{}_Gamma_reweight_r_{}_g_{}_{}_4comp_1copy_2stage_full".format(dataset, repeat, gamma_b, gamma_c).replace('.', '')
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    filename = '/'.join(['.', result_dir, filename])
    
    for rep in range(num_rep):
        print("Repetition: {}".format(rep))
        for i, copy_rate_value in enumerate(copy_rate_range):
            copy_rates[copy_ids] = copy_rate_value
            print("----------------")
            print("Copy rates: {}".format(copy_rates))
            conf_b = generate_conf_pairflipper(num_busy, k, gamma_b)
            conf = generate_conf_pairflipper(m, k, gamma_c)
            conf[:num_busy] = conf_b
            labels_train, _, workers_on_example = generate_labels_weight_sparse_copycat(y_train[valid_range], repeat, conf, copy_rates, num_busy)
            labels_vali, _, workers_on_example_vali = generate_labels_weight_sparse_copycat(y_vali, repeat, conf, copy_rates, num_busy)
            
            # 1. pretrain on majority vote
            print("Pretraining model on weighted majority vote...")
            y_train_wmv = np.sum(labels_train, axis=1) / repeat
            y_vali_corrupt = np.sum(labels_vali, axis=1) / repeat
            pred_train, pred_vali, vali_acc, test_acc, init_model = call_train_mbem(X_train, valid_range, y_train_wmv, X_vali, y_vali_corrupt, y_vali, X_test, y_test, use_aug=use_aug, dataset=dataset)
            test_accs[rep, i, 2] = test_acc
            # conf_errors[rep, i, 2] = conf_error # not applicable
            # cp_errors[rep, i, 2] = cp_error # not applicable
            # _, _, est_conf = posterior_distribution(labels_train, pred_train, workers_on_example)
            # plot_conf_mat(est_conf, conf)

            # 2. Reweighting according to gamma + both label counts and estimated copy probs
            print("--------")
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=True, use_pretrained=True, model=init_model, conf_init=None, use_aug=use_aug, est_cr=True, reweight="GAMMA+BOTH", dataset=dataset)
            test_accs[rep, i, 0] = test_acc
            conf_errors[rep, i, 0] = conf_error
            cp_errors[rep, i, 0] = cp_error
            print(est_copyrates[1:])
            # plot_conf_mat(est_conf, conf)
            
            # 3. w/o copy rate est.
            print("--------")
            est_conf, est_copyrates, test_acc, conf_error, cp_error = call_train(X_train, valid_range, labels_train, X_vali, labels_vali, y_vali, X_test, y_test, 
                                                            conf, copy_rates, two_stage=False, use_pretrained=False, model=None, use_aug=use_aug, est_cr=False, reweight=False, dataset=dataset)
            test_accs[rep, i, 1] = test_acc
            conf_errors[rep, i, 1] = conf_error
            # cp_errors[rep, i, 1] = cp_error # not applicable
            
            # 4. MBEM
            print("--------")
            for j in range(mbem_round):
                est_q, est_label_posterior, est_conf = posterior_distribution(labels_train, pred_train, workers_on_example)
                est_q_vali, est_label_posterior_vali, _ = posterior_distribution(labels_vali, pred_vali, workers_on_example_vali)
                # Train
                pred_train, pred_vali, vali_acc, test_acc, model = call_train_mbem(X_train, valid_range, est_label_posterior, X_vali, est_label_posterior_vali, y_vali, X_test, y_test, use_aug=use_aug, dataset=dataset)
            conf_error = np.mean([np.linalg.norm(est_conf[i] - conf[i]) for i in range(m)])
            test_accs[rep, i, 3] = test_acc
            conf_errors[rep, i, 3] = conf_error
            # cp_errors[rep, i, 3] = cp_error # not applicable
            # plot_conf_mat(est_conf, conf)
            
            
    plot_result_std_cr(test_accs, copy_rate_range, title=title, ylabel="Test accuracy", filename=filename+"_testacc")
    plot_result_std_cr(cp_errors, copy_rate_range, title=title, ylabel="Copy probability estimation error", filename=filename+"_cperror")
    plot_result_std_cr(conf_errors, copy_rate_range, title=title, ylabel="Confusion matrix estimation error", filename=filename+"_conferror")
