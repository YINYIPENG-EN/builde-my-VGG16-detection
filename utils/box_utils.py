import numpy as np
import torch
from PIL import Image

def point_form(boxes):
    #------------------------------#
    #   获得框的左上角和右下角
    #------------------------------#
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                     boxes[:, :2] + boxes[:, 2:]/2), 1)

def center_size(boxes):
    #------------------------------#
    #   获得框的中心和宽高
    #------------------------------#
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,
                     boxes[:, 2:] - boxes[:, :2], 1)

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    #------------------------------#
    #   获得交矩形的左上角
    #------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    #------------------------------#
    #   获得交矩形的右下角
    #------------------------------#
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)
    #-------------------------------------#
    #   计算先验框和所有真实框的重合面积
    #-------------------------------------#
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    #-------------------------------------#
    #   返回的inter的shape为[A,B]
    #   代表每一个真实框和先验框的交矩形
    #-------------------------------------#
    inter = intersect(box_a, box_b)
    #-------------------------------------#
    #   计算先验框和真实框各自的面积
    #-------------------------------------#
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    #-------------------------------------#
    #   每一个真实框和先验框的交并比[A,B]
    #-------------------------------------#
    return inter / union


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    #----------------------------------------------#
    #   计算所有的先验框和真实框的重合程度
    #----------------------------------------------#
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    #----------------------------------------------#
    #   所有真实框和先验框的最好重合程度
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,0]
    #----------------------------------------------#
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    #----------------------------------------------#
    #   所有先验框和真实框的最好重合程度
    #   best_truth_overlap [1,prior]
    #   best_truth_idx [1,prior]
    #----------------------------------------------#
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    #----------------------------------------------#
    #   用于保证每个真实框都至少有对应的一个先验框
    #----------------------------------------------#
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    
    #----------------------------------------------#
    #   获取每一个先验框对应的真实框[num_priors,4]
    #----------------------------------------------#
    matches = truths[best_truth_idx]
    # Shape: [num_priors]
    conf = labels[best_truth_idx] + 1

    #----------------------------------------------#
    #   如果重合程度小于threhold则认为是背景
    #----------------------------------------------#
    conf[best_truth_overlap < threshold] = 0
    
    #----------------------------------------------#
    #   利用真实框和先验框进行编码
    #   编码后的结果就是网络应该有的预测结果
    #----------------------------------------------#
    loc = encode(matches, priors, variances)

    # [num_priors,4]
    loc_t[idx] = loc    
    # [num_priors]
    conf_t[idx] = conf  

def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1) 

#-------------------------------------------------------------------#
#   Adapted from https://github.com/Hakuyume/chainer-ssd
#   利用预测结果对先验框进行调整，前两个参数用于调整中心的xy轴坐标
#   后两个参数用于调整先验框宽高
#-------------------------------------------------------------------#
def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],  # 调整先验框中心的位置 loc[:,:2]回归预测结果前两位的结果，priors[:, 2:]先验框的长和宽
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)   # 调整后先验框的宽和高
    boxes[:, :2] -= boxes[:, 2:] / 2  # 先验框的左上角
    boxes[:, 2:] += boxes[:, :2]   # 先验的右下角
    return boxes


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

#-------------------------------------------------------------------#
#   Original author: Francisco Massa:
#   https://github.com/fmassa/object-detection.torch
#   Ported to PyTorch by Max deGroot (02/01/2017)
#   该部分用于进行非极大抑制，即筛选出一定区域内得分最大的框。
#-------------------------------------------------------------------#
def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union 
        idx = idx[IoU.le(overlap)]
    return keep, count

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def ssd_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2, (left+right)/2),axis=-1)
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes
