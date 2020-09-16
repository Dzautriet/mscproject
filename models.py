# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:40:47 2020

@author: Zhe Cao
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=(1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*3*3, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = self.logsoftmax(x)
        return out
        
    
class CNN_CIFAR(nn.Module):
    """
    ResNet Wrapper
    """
    def __init__(self, precnn, num_classes=10):
        super(CNN_CIFAR, self).__init__()
        self.last_in = precnn.fc.in_features # 512 for ResNet18, 2048 for ResNet50
        self.precnn = nn.Sequential(*list(precnn.children())[:-1])
        self.fc = nn.Linear(self.last_in, num_classes, bias=True) # Replace the last fc
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.precnn(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)
        output = self.logsoftmax(output)
        return output
    

class ConfMatLayer(nn.Module):
    """
    Adapted for ablation study
    """
    def __init__(self, m, k, est_cr=True, reweight=False, factor=None, conf_init=None):
        """
        m: number of total labellers
        k: number of classes
        est_cr: whether estimate copy rates or not
        reweight: whether reweight losses according to copy rates and label counts or not
        factor: reweighting factor for label count imbalance
        
        """
        super(ConfMatLayer, self).__init__()
        self.m = m
        self.est_cr = est_cr # Whether estimate copy rates or not
        self.reweight = reweight # Whether reweight losses according to copy rates and label counts or not
        self.factor = factor # reweighting factor that accounts for sparsity
        if conf_init is None:
            b_init = torch.tensor(np.stack([6.*np.eye(k)-5. for i in range(m)]), dtype=torch.float32)
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
        """
        Calculate confusion matrices
        """
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
                elif self.reweight == "BOTH":
                    # losses_weight[0] -= self.copyrates[1:].sum()
                    losses_weight[0] -= torch.clamp(self.copyrates[1:].sum(), 0, 1)
                    losses_weight[0] *= self.factor
                elif self.reweight == "GAMMA+CNT":
                    est_gamma = torch.tensor([torch.diag(self.confusion_matrices[i]).mean() for i in range(self.m)]).to(self.b.device)
                    factor_gamma = est_gamma[0] / (est_gamma[1:].mean())
                    if factor_gamma < 1.5:
                        losses_weight[0] *= self.factor
                elif self.reweight == "GAMMA+CP":
                    est_gamma = torch.tensor([torch.diag(self.confusion_matrices[i]).mean() for i in range(self.m)]).to(self.b.device)
                    factor_gamma = est_gamma[0] / (est_gamma[1:].mean())
                    if factor_gamma < 1.5:
                        losses_weight[0] -= torch.clamp(self.copyrates[1:].sum(), 0, 1)
                else:
                    raise ValueError("Illegal value for reweight!")
                losses_weight = torch.clamp(losses_weight, 1e-3)
            losses_all_users *= losses_weight
        
        losses_all_users = torch.mean(torch.sum(losses_all_users, axis=1))
        return losses_all_users