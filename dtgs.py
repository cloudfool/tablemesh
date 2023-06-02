"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
#from collections import OrderedDict
#import torch.nn.init as init
from torchutil import *
#from deform_conv_v2 import *
#import os
from basenet.vgg16_bn import vgg16_bn
#import timm
import numpy as np
#from efficientnet_pytorch import EfficientNet
#from cc_attention import CrissCrossAttention
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import backbones
#from pprint import pprint
class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

##pyramid pooling module(PPM)
#replace adaptiveAvgPool with AvgPool
class PPM_1(nn.Module):
    def __init__(self, in_dim, reduction_dim):
        super(PPM_1, self).__init__()
        self.features = []
        for i in range(4):
            self.features.append(nn.Sequential(
                #nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()#b,c,w,h
        out = [x]
        avgs = []
        bins = (1,2,3,6)
        inputsz = np.array(x_size[2:])
        for bin in bins:
            outputsz = np.array([bin,bin])
            stridesz = np.floor(inputsz/outputsz).astype(np.int32)
            kernelsz = inputsz-(outputsz-1)*stridesz
            avgs.append(nn.AvgPool2d(kernel_size=list(kernelsz),stride=list(stridesz)))

        for avg,f in zip(avgs,self.features):
            out.append(F.interpolate(f(avg(x)), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class DTGS_1(nn.Module):
    def __init__(self, pretrained=True, freeze=False,direction=0):
        super(DTGS_1, self).__init__()

        """ Base network """

        count = 1 #appm need set 2
        #self.basenet = vgg16_bn(pretrained, freeze)

        self.basenet = getattr(backbones, 'resnet18')()
        """ U network """
        self.upconv1 = double_conv(512, 256, 128)
        self.upconv2 = double_conv(128, 128, 64)
        self.upconv3 = double_conv(64, 64, 32*count)

        fea_dim = 32*count

        self.M = PPM_1(fea_dim, int(fea_dim/4))
        out_fea_dim = 32*2*count


        init_weights(self.M.modules())

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(out_fea_dim, 32*count, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32*count, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())

        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """

        y = F.interpolate(sources[3], size=sources[2].size()[2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv1(y)


        y = F.interpolate(y, size=sources[1].size()[2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, sources[1]], dim=1)
        y = self.upconv2(y)


        y = F.interpolate(y, size=sources[0].size()[2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, sources[0]], dim=1)
        feature = self.upconv3(y)
        feature = F.interpolate(feature, scale_factor=2, mode='bilinear')


        feature = self.M(feature)

        y = self.conv_cls(feature)

        #print ('sb', y.shape, feature.shape)

        return y.permute(0, 2, 3, 1), feature



if __name__ == '__main__':

    #model_names = timm.list_models(pretrained=True)
    #pprint(model_names)

    model = DTGS_1(pretrained=True).cuda()
    output, _ = model(torch.randn(2, 3, 768, 1024).cuda())
    print(output.shape)
