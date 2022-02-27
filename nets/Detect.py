from __future__ import division
from math import sqrt as sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from utils.box_utils import decode, nms
from utils.config import Config

#  解码预测过程
class Detect(nn.Module):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        super().__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = Config['variance']


    def forward(self, loc_data, conf_data, prior_data):
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()


        num = loc_data.size(0)  # shape is batch_size
        num_priors = prior_data.size(0)  # 先验框数量

        output = torch.zeros(num, self.num_classes, self.top_k, 5)  # 建立一个output阵列存放输出
        #--------------------------------------#
        #   对分类预测结果进行reshape
        #   num, num_classes, num_priors
        #--------------------------------------#
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1) # transpose是将num_priors,num_classes维度进行反转

        # 对每一张图片进行处理正常预测的时候只有一张图片，所以只会循环一次
        for i in range(num):
            #--------------------------------------#
            #   对先验框解码获得预测框
            #   解码后，获得的结果的shape为
            #   num_priors, 4
            #--------------------------------------#
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # 回归预测loc_data的结果对先验框进行调整
            conf_scores = conf_preds[i].clone()

            #--------------------------------------#
            #   获得每一个类对应的分类结果
            #   num_priors,
            #--------------------------------------#
            for cl in range(1, self.num_classes):
                #--------------------------------------#
                #   首先利用门限进行判断
                #   然后取出满足门限的得分
                #--------------------------------------#
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                #--------------------------------------#
                #   将满足门限的预测框取出来
                #--------------------------------------#
                boxes = decoded_boxes[l_mask].view(-1, 4)
                #--------------------------------------#
                #   利用这些预测框进行非极大抑制
                #--------------------------------------#
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
        return output


