YOLO_v2
=
## Usage
1. clone YOLO_v2 repository
``` bash
git clone https://github.com/Stinky-Tofu/YOLO_v2.git
```
2. Download dataset <br>
Create a new folder named data in the directory where the YOLO folder is located, and then create a new folder named Pascal_voc in the data folder.<br>
Download [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [Pascal VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) dataset，and then put the two data sets in the Pascal Voc folder, name VOC2012 as VOCdevkit, and name VOC2007 as VOCdevkit-test <br>
3. Download pre-trained model<br>
Download the model trained on coco datasets[yolo_coco_initial.ckpt](https://drive.google.com/drive/folders/19m9KpAmBP1GTGvC2x5XCSvsDW-psEXF5?hl=zh-CN)(This model pre-trained on the coco dataset and then modifies the last convolutional layer to apply to the Pascal_VOC dataset.), then put this model in the YOLO/model/.
4. Train<br>
``` bash
python train.py
--model_file(The name of the model under the model folder, the default is yolo_coco_initial.ckpt)
--gpu(Gpu used during training, the default is 0,1)
```
5. Test<br>
``` bash
python test.py
--model_file(The name of the model under the model folder, the default is None, required.)
--video_path(Path to the video used for testing, the default is None, optional)
--image_path(Path to the image used for testing, the default is ./data/original/car.jpg, optional)
--video_save_path(Video save path, the default is None, optinal)
--image_save_path(Image save path, the default is None, optinal)
--gpu(Gpu used during testing，the default is None，optinal)
```
## requirements
. Tensorflow <br>
. Opencv2 <br>
