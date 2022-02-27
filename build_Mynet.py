import torch
import torch.nn as nn
from nets.vgg import vgg
from torchsummary import summary
from math import sqrt as sqrt
import numpy as np
from nets.Detect import Detect
from torch.autograd import Variable
from utils.config import Config
import matplotlib.pyplot as plt
class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # 获得输入图片的大小，默认为300x300
        self.image_size = 300
        self.num_priors = 1  # 先验框数量
        self.variance = [0.1]
        self.feature_maps = [19]  # conv7特征层大小
        self.min_sizes = [60]  # 对应最小的先验框尺寸
        self.max_sizes = [111]  # # 对应最大的先验框尺寸
        self.steps = [16]  # 相当于该特征层和原图的映射关系
        self.aspect_ratios = [[2]]
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # ----------------------------------------#
        #   获得1个大小为19*19的有效特征层用于预测
        # ----------------------------------------#
        for k, f in enumerate(self.feature_maps):
            #----------------------------------------#
            #   对特征层生成网格点坐标矩阵
            #----------------------------------------#
            x,y = np.meshgrid(np.arange(f),np.arange(f))
            x = x.reshape(-1)
            y = y.reshape(-1)

            '''plt.plot(x,y,color='red',marker='.',linestyle='')
            plt.grid(True)
            plt.show()'''

            #----------------------------------------#
            #   所有先验框均为归一化的形式
            #   即在0-1之间
            #----------------------------------------#
            for i, j in zip(y,x):
                f_k = self.image_size / self.steps[k]
                #----------------------------------------#
                #   计算网格的中心
                #----------------------------------------#
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                #----------------------------------------#
                #   获得小的正方形
                #----------------------------------------#
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                #----------------------------------------#
                #   获得大的正方形
                #----------------------------------------#
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                #----------------------------------------#
                #   获得两个的长方形
                #----------------------------------------#
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        #----------------------------------------#
        #   获得所有的先验框 8732,4
        #----------------------------------------#
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class Mymodel(nn.Module):
    def __init__(self, phase, num_classes,confidence, nms_iou):
        super(Mymodel, self).__init__()
        self.phase = phase
        self.cfg = Config
        self.num_classes = num_classes
        self.backbone = nn.ModuleList(vgg(3))
        box = [4]  # 最后一个特征层锚框数量
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        loc_layers = [nn.Conv2d(self.backbone[-2].out_channels, box[0] * 4, kernel_size=3,padding=1)]
        conf_layers = [nn.Conv2d(self.backbone[-2].out_channels, box[0] * num_classes, kernel_size = 3, padding = 1)]

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)  # 所有类的概率相加为1
            # Detect(num_classes,bkg_label,top_k,conf_thresh,nms_thresh)
            # top_k:一张图片中，每一类的预测框的数量
            # conf_thresh 置信度阈值
            # nms_thresh：值越小表示要求的预测框重叠度越小，0.0表示不允许重叠
            self.detect = Detect(num_classes, 0, 200, confidence, nms_iou)

    def forward(self, x):
        loc = list()
        conf = list()
        featrue = list()
        for k in range(len(self.backbone)):
            x = self.backbone[k](x)
        featrue.append(x)  # 最后一个特征层为batch_size,1024,19,19

        for (x, l, c) in zip(featrue, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output

