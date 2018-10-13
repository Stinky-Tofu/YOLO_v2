YOLO_v2
=
## Usage
1. clone YOLO_v2 repository
``` bash
git clone https://github.com/Stinky-Tofu/YOLO_v2.git
```
2. 下载数据 <br>
在YOLO所在目录新建一个data文件夹，然后在data文件夹下新建一个Pascal_voc文件夹<br>
下载[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)和[Pascal VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)数据集，然后将这两个数据集放在Pascal_voc文件夹下，并将VOC2012命名为VOCdevkit，将VOC2007命名为VOCdevkit-test <br>
3. 下载预训练模型<br>
下载在coco数据上训练过的模型（这个模型首先在COCO数据集上预训练过，然后修改最后一个卷积层，使之适用于Pascal_VOC数据集），然后将这个模型放入YOLO/model/文件夹下。
4. 训练<br>
``` bash
python train.py
--model_file(model文件夹下模型的名字，默认为yolo_coco_initial.ckpt)
--gpu(训练时使用的gpu，默认为0,1)
```
5. 测试<br>
``` bash
python test.py
--model_file(model文件夹下模型的名字，默认为None)
--video_path(用于测试的video的路径，默认为None)
--image_path(用于测试的image的路径，默认为None)
--video_save_path(测试后的video的保存路径, 默认为./data/original/car.jpg)
--image_save_path(测试后的image的保存路径，默认为None)
--gpu(测试时使用的gpu，默认为None)
```
## requirements
. Tensorflow <br>
. Opencv2 <br>
