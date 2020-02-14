# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:40:47 2020

@author: Zhe Cao
"""
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
    
# model = CNN_MNIST()
# X = torch.ones((128, 1, 28, 28))
# y = model(X)