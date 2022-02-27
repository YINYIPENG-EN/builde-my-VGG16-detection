from IPython import embed
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from build_Mynet import Mymodel
import torch.optim as optim
from Datasets import MyDatasets, my_dataset_collate
from nets.training import MultiBoxLoss
from utils.config import Config
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def do_train(model, criterion, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    loc_loss = 0
    conf_loss = 0
    loc_loss_val = 0
    conf_loss_val = 0
    model.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]  # images[batch_size,3,300,300]
            # targets是一个列表，有5个向量(x1,y1,x2,y2,标签)
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]

                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            output = model(
                images)  # output返回loc(batch_size,num_anchors,4[坐标])，conf(batch_size,num_anchors,num_classes)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(output, targets)
            loss = loss_l + loss_c
            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            pbar.set_postfix(**{'loc_loss': loc_loss / (iteration + 1),
                                'conf_loss': conf_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    model.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

                out = model(images)
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)

                loc_loss_val += loss_l.item()
                conf_loss_val += loss_c.item()

                pbar.set_postfix(**{'loc_loss': loc_loss_val / (iteration + 1),
                                    'conf_loss': conf_loss_val / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    total_loss = loc_loss + conf_loss
    val_loss = loc_loss_val + conf_loss_val
    print('\nFinish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))

    torch.save(model.state_dict(),r'./logs/mymodel.pth')

if __name__ == '__main__':
    cuda = True
    criterion = MultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, cuda)  # 定义损失函数
    model = Mymodel("train", Config['num_classes'], confidence=0.6, nms_iou=0.5)
    model_path = r'./model_data/Mymodel.pth'
    model_dict = model.state_dict()
    device = torch.device('cuda')
    pretrained_dict = torch.load(model_path, map_location=device)
    #pretrained_dict = {k:v for k,v in pretrained_dict.items() if np.shape(model_dict[k])== np.shape(pretrained_dict[k])}
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if pretrained_dict.keys()==model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("完成预权重的加载")
    model.to(device)
    batch_size = 8

    annotation_path = r'2007_train.txt'
    with open(annotation_path, encoding='utf-8') as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    val = 0.1
    num_val = int(len(lines) * val)
    num_train = len(lines) - num_val

    model.train()


    Use_Data_Loader = True

    lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    Init_Epoch = 0
    Freeze_Epoch = 50

    if Use_Data_Loader:
        train_dataset = MyDatasets(lines[:num_train], (300, 300),
                                   True)  # train_dataset返回数据集和标签(且这个是可以迭代的)
        gen = DataLoader(train_dataset, batch_size, False, num_workers=4, pin_memory=True,
                         collate_fn=my_dataset_collate)
        val_dataset = MyDatasets(lines[num_train:], (300, 300), False)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=my_dataset_collate)
        epoch_size = num_train // batch_size
        epoch_size_val = num_val // batch_size

        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(50):
            do_train(model, criterion, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, cuda)
            lr_scheduler.step()
