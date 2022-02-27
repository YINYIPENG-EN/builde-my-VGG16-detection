import colorsys
import os
import warnings

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from build_Mynet import Mymodel
from utils.box_utils import letterbox_image, ssd_correct_boxes
from utils.config import Config

warnings.filterwarnings("ignore")

MEANS = (104, 117, 123)


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的config里面的num_classes、
#   model_path和classes_path参数的修改
# --------------------------------------------#
class VGG16(object):
    _defaults = {

        "model_path": 'logs/mymodel.pth',
        "classes_path": 'model_data/new_classes.txt',
        "input_shape": (300, 300, 3),
        "confidence": 0.6,
        "nms_iou": 0.5,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   计算总的类的数量
        # -------------------------------#
        self.num_classes = len(self.class_names) + 1

        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#

        model = Mymodel("test",Config['num_classes'],self.confidence, self.nms_iou)
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()


        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):

        image_shape = np.array(np.shape(image)[0:2])

        # ---------------------------------------------------#
        #   不失真的resize，给图像周围增加灰条
        # ---------------------------------------------------#
        crop_img = np.array(letterbox_image(image, (self.input_shape[1], self.input_shape[0])))

        with torch.no_grad():
            # ---------------------------------------------------#
            #   图片预处理，归一化
            # ---------------------------------------------------#
            photo = Variable(
                torch.from_numpy(np.expand_dims(np.transpose(crop_img - MEANS, (2, 0, 1)), 0)).type(torch.FloatTensor))
            if self.cuda:
                photo = photo.cuda()

            # ---------------------------------------------------#
            #   传入网络进行预测
            # ---------------------------------------------------#
            preds = self.net(photo)

            top_conf = []
            top_label = []
            top_bboxes = []
            # ---------------------------------------------------#
            #   preds的shape为 1, num_classes, top_k, 5
            # ---------------------------------------------------#
            for i in range(preds.size(1)):
                j = 0
                while preds[0, i, j, 0] >= self.confidence:
                    # ---------------------------------------------------#
                    #   score为当前预测框的得分
                    #   label_name为预测框的种类
                    # ---------------------------------------------------#
                    score = preds[0, i, j, 0]
                    label_name = self.class_names[i - 1]
                    # ---------------------------------------------------#
                    #   pt的shape为4, 当前预测框的左上角右下角
                    # ---------------------------------------------------#
                    pt = (preds[0, i, j, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]
                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    j = j + 1

        # 如果不存在满足门限的预测框，直接返回原图
        if len(top_conf) <= 0:
            return image

        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1],
                                                                                                      -1), np.expand_dims(
            top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        # -----------------------------------------------------------#
        #   去掉灰条部分
        # -----------------------------------------------------------#
        boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                  np.array([self.input_shape[0], self.input_shape[1]]), image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        for i, c in enumerate(top_label):
            predicted_class = c
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))



            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

