# -VGG16-
实现从零开始，搭建自己的目标检测网络，实现训练和预测：https://blog.csdn.net/z240626191s/article/details/123160343

数据集处理：继承Dataset处理自己的数据教程，可参考：https://blog.csdn.net/z240626191s/article/details/123108750

通过Dataloader加载上述已处理的数据集教程：https://blog.csdn.net/z240626191s/article/details/123129630

仿照SSD网络定义自己的目标检测网络，选择VGG16为目标检测网络，先验框的计算和解码方式均采用SSD中的方法
本代码参考于：https://blog.csdn.net/weixin_44791964/article/details/104981486?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164596303916780357223568%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164596303916780357223568&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-5-104981486.pc_search_result_positive&utm_term=%E7%9D%BF%E6%99%BASSD&spm=1018.2226.3001.4187

使用说明：
从百度网盘下载权重：
链接：https://pan.baidu.com/s/1tS8W1fsjOJEiE8yDUmlrcw 
提取码：yypn

权重下载以后放入logs文件夹，运行predict.py，输入test.jpg即可出检测结果

训练自己的数据集：

1.如果训练自己的数据集【VOC格式】,图像放在VOCdevkit/VOC2007/JPEGImages/，xml标签放在VOCdevkit/VOC2007/Annotations/

2.运行voc2vgg.py将会在ImageSets下生成txt文件

3.修改voc_annotation.py中的类名字(和自己的类名字一样)后运行，将会生成2007_train.txt等文件

4.修改utils中的config.py文件中num_classes，注意类的数量为自己的数量+1(因为有个背景类)

5.在model_data中的new_classes.txt添加自己的类名

6.运行Mymodel_train.py【可以修改epoch，batch_size等参数】

训练完成后会在logs保存权重


详细搭建过程可参考https://blog.csdn.net/z240626191s/article/details/123160343  包括网络搭建，先验框的计算，训练，预测等。
