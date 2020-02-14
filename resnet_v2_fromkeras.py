# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:17:16 2020

@author: Zhe Cao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation="relu", batch_normalization=True, conv_first=True):
    conv = nn.Conv2d(in_channels=)
    x = inputs
    
    return x

def resnet(input_shape, depth, num_classes=10):
    

class ResNet(nn.Module):
    def __init__(self, input_shape, depth, num_classes=10):
        super(ResNet, self).__init__()
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)
        
            
    def _resnet_layer(self, x):
        
        
    def forward(self, x):
        
        