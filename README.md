YOLO_v2
=
安装
-
1. clone YOLO_v2 repository
``` bash
git clone https://github.com/Stinky-Tofu/YOLO_v2.git
```
2. 下载数据 <br>
在YOLO所在目录新建一个data文件夹，然后在data文件夹下新建一个Pascal_voc文件夹<br>
下载[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)和[Pascal VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)数据集，然后将这两个数据集放在Pascal_voc文件夹下，并将VOC2012命名为VOCdevkit，将VOC2007命名为VOCdevkit-test <br>
3. 下载预训练模型<br>
下载在coco数据上训练过的模型，已将这个coco模型的最后一个卷积层已被
