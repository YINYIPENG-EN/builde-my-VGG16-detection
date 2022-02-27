import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from IPython import embed

MEANS = (104, 117, 123)


class MyDatasets(Dataset):
    def __init__(self, train_line, image_size, is_train):
        super(MyDatasets, self).__init__()
        self.train_line = train_line
        self.train_batches = len(train_line)
        self.image_size = image_size
        self.is_train = is_train
        # embed()

    def get_data(self, annotation_line, input_shape, random=True):

        line = annotation_line.split()
        image = Image.open(line[0])  # line[0]是图片路径，line[1:]是框和标签信息
        iw, ih = image.size  # 真实输入图像大小
        h, w = input_shape  # 网络输入大小
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])  # 将box信息转为数组
        if not random:
            # 裁剪图像
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2  # 取商（应该是留部分条状）
            dy = (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC) # 采用双三次插值算法缩小图像
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 处理真实框
            box_data = np.zeros((len(box), 5))
            if (len(box) > 0):
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx  # 对原框y坐标缩放
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy  # 对原x坐标进行缩放

                # 处理左上坐标,防止负坐标
                box[:, 0:2][box[:, 0:2] < 0] = 0

                # 处理右下坐标，防止超过输入边界
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h

                # 计算缩放后的框的尺寸
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]

                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            return image_data, box_data

    def __len__(self):  # 返回数据集的长度

        return self.train_batches

    def __getitem__(self, index):  # 返回数据集和标签
        lines = self.train_line

        if self.is_train:
            img, y = self.get_data(lines[index], self.image_size[0:2], random=False)
        else:
            img, y = self.get_data(lines[index], self.image_size[0:2], random=False)

        boxes = np.array(y[:, :4], dtype=np.float32)

        boxes[:, 0] = boxes[:, 0] / self.image_size[1]
        boxes[:, 1] = boxes[:, 1] / self.image_size[0]
        boxes[:, 2] = boxes[:, 2] / self.image_size[1]
        boxes[:, 3] = boxes[:, 3] / self.image_size[0]

        boxes = np.maximum(np.minimum(boxes, 1), 0)
        y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img - MEANS, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)

        return tmp_inp, tmp_targets

def my_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes

