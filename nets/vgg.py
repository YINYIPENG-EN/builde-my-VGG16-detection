import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# 从列表中循环通道，M和C均为最大池化选项
base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512]

def vgg(i):
    layers = []  # 用于存放vgg网络的list
    in_channels = i  # 最前面那层的维度--300*300*3，因此i=3
    for v in base:  # 在通道列表中进行遍历
        if v == 'M':  # 池化，补不补边界
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:  # 判断是否在池化层，卷积层都是3*3，v是输出通道数
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # 每层卷积后使用ReLU激活函数
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # 通道（1024）膨胀，用卷积层代替全连接层
    # #  dilation=卷积核元素之间的间距,扩大卷积感受野的范围，没有增加卷积size,使用了空洞数为6的空洞卷积
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)  # 在conv7输出19*19的特征层
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    # layers有20层 VGG前17层，加上pool5,conv6,conv7
    return layers
