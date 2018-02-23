import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import math
import numpy as np
from LAF import extract_patches
from architectures import LocalNorm2d

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-8
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x = x / norm.unsqueeze(1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x


class HardTFeatNet(nn.Module):
    """TFeat model definition
    """

    def __init__(self, sm):
        super(HardTFeatNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=8),
            nn.Tanh())
        self.SIFT = sm
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        x = self.classifier(x_features)
        return x.view(x.size(0), -1)

class HardNetNarELU(nn.Module):
    """TFeat model definition
    """

    def __init__(self,sm):
        super(HardNetNarELU, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=8),
            nn.BatchNorm2d(128, affine=False))
        self.SIFT = sm
        return

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        #print(sp)
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(input)#self.input_norm(input))
        #x = self.classifier[1](x_features)
        x = nn.AdaptiveAvgPool2d(1)(x_features).view(x_features.size(0), -1)
        return x
        #return L2Norm()(x)


class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
        #self.features.apply(weights_init)

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x = self.features(self.input_norm(input))
        return  L2Norm()(nn.AdaptiveMaxPool2d(1)(x).view(-1,128))
class HardNetConv(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNetConv, self).__init__()
        self.lrn = LocalNorm2d(15)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
            L2Norm()
        )
        return
    def forward(self, input):
        return self.features(self.lrn(input))
class HardNetConvCrop(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNetConvCrop, self).__init__()
        self.lrn = LocalNorm2d(15)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
        )
        return
    def load_from_trained_HardNet(self,fname):
        pretrained_dict = torch.load(fname)['state_dict']
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        #model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)
        return
    def forward(self, input):
        return self.features(self.lrn(input))
    
class HardNetHead(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNetHead, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
            L2Norm()
        )
        return
    def load_from_trained_HardNet(self,fname):
        pretrained_dict = torch.load(fname)['state_dict']
        self.features[0].weight.data = pretrained_dict['features.12.weight']
        self.features[1].running_mean = pretrained_dict['features.13.running_mean']
        self.features[1].running_var = pretrained_dict['features.13.running_var']
        self.features[3].weight.data = pretrained_dict['features.15.weight']
        self.features[4].running_mean = pretrained_dict['features.16.running_mean']
        self.features[4].running_var = pretrained_dict['features.16.running_var']
        self.features[6].weight.data = pretrained_dict['features.19.weight']
        self.features[7].running_mean = pretrained_dict['features.20.running_mean']
        self.features[7].running_var = pretrained_dict['features.20.running_var']
        return
    def forward(self, input):
        return self.features(input)
class DeformedHardNetHead(nn.Module):
    """HardNet model definition
    """
    def __init__(self, desc):
        super(DeformedHardNetHead, self).__init__()
        self.HNHead = desc
        return
    def forward(self, cropped_feats, nLAFs):
        return self.HNHead(extract_patches(cropped_feats, nLAFs, PS = 16))