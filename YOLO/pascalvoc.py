# coding: utf-8

import numpy as np
import os
import config as cfg
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import math
import cv2


class PascalVoc(object):
    def __init__(self, model):
        self.__pascal_voc = os.path.join(cfg.DATA_DIR, 'Pascal_voc')
        self.__image_size = cfg.IMAGE_SIZE
        self.__cell_size = cfg.CELL_SIZE
        self.__down_sample_size = 1.0 * self.__image_size / self.__cell_size
        self.__batch_size = cfg.BATCH_SIZE
        self.__classes = cfg.CLASSES
        self.__num_classes = len(self.__classes)
        self.__box_per_cell = cfg.BOX_PRE_CELL
        self.__class_to_ind = dict(zip(self.__classes, range(self.__num_classes)))
        self.__anchor = cfg.ANCHOR

        self.__samples = self.__load_samples(model)
        self.__num_samples = len(self.__samples)
        self.__num_batchs = math.ceil(self.__num_samples / self.__batch_size)
        self.__count = 0

    def __load_samples(self, model):
        """
        :param model: 选择加载训练样本或测试样本，必须是'train' or 'test'
        :return: model对应的样本
        """
        if model == 'train':
            self.__devkit_path = os.path.join(self.__pascal_voc, 'VOCdevkit')
            self.__data_path = os.path.join(self.__devkit_path, 'VOC2012')
            txtname = os.path.join(self.__data_path, 'ImageSets', 'Main', 'trainval.txt')
        elif model == 'test':
            self.__devkit_path = os.path.join(self.__pascal_voc, 'VOCdevkit-test')
            self.__data_path = os.path.join(self.__devkit_path, 'VOC2007')
            txtname = os.path.join(self.__data_path, 'ImageSets', 'Main', 'test.txt')
        else:
            raise ImportError("You must choice one of the 'train' or 'test' for model parameter")

        with file(txtname, 'r') as f:
            txt = f.readlines()
            image_ind = [line.strip() for line in txt]

        samples = []
        for ind in image_ind:
            label, num = self.__load_label(ind)
            if num == 0:
                continue
            image_name = os.path.join(self.__data_path, 'JPEGImages', ind + '.jpg')
            samples.append({'image_name': image_name, 'yolo_target': label})
        np.random.shuffle(samples)
        return samples

    def __load_label(self, index):
        """
        :param index: index为标签索引号
        :return: index对应的标签，标签的shape为(cell_size, cell_size, box_per_cell, 5+num_classes)
        如果物体落入某个grid，那么这个grid对应的所有Anchor的标签相同，这些Anchor的标签为confidence, x,y,w,h, classes
        """
        filename = os.path.join(self.__data_path, 'Annotations', index + '.xml')
        root = ET.parse(filename).getroot()
        image_size = root.find('size')
        image_width = float(image_size.find('width').text)
        image_height = float(image_size.find('height').text)
        h_ratio = 1.0 * self.__image_size / image_height
        w_ratio = 1.0 * self.__image_size / image_width

        # 只保存GT的的中心坐标、高宽、类别
        label = np.zeros((self.__cell_size, self.__cell_size, self.__box_per_cell, 5 + self.__num_classes))
        objects = root.findall('object')
        for obj in objects:
            box = obj.find('bndbox')
            xmin = max(min(float(box.find('xmin').text) * w_ratio, self.__image_size), 0)
            xmax = max(min(float(box.find('xmax').text) * w_ratio, self.__image_size), 0)
            ymin = max(min(float(box.find('ymin').text) * h_ratio, self.__image_size), 0)
            ymax = max(min(float(box.find('ymax').text) * h_ratio, self.__image_size), 0)

            # grid在图上的大小为down_sample_size，此处得到GT相对于down_sample_size大小的中心坐标(x,y)、宽高(w,h)
            coor = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, xmax - xmin, ymax - ymin]) / self.__down_sample_size
            xind = int(np.floor(coor[0]))
            yind = int(np.floor(coor[1]))
            class_ind = self.__class_to_ind[obj.find('name').text.lower().strip()]

            label[yind, xind, :, 0] = 1
            label[yind, xind, :, 1:5] = coor
            label[yind, xind, :, 5 + class_ind] = 1

        return label, len(objects)

    def __iter__(self):
        return self

    def __len__(self):
        return self.__num_batchs

    def next(self):
        """
        :return: 使得pascal_voc类变为可迭代对象，每次迭代返回一个batch的样本、标签
        """
        images = np.zeros((self.__batch_size, self.__image_size, self.__image_size, 3))
        labels = np.zeros((self.__batch_size, self.__cell_size, self.__cell_size,
                           self.__box_per_cell, 5 + self.__num_classes))
        num = 0
        if self.__count < self.__num_batchs:
            while num < self.__batch_size:
                index = self.__count * self.__batch_size + num
                if index >= self.__num_samples:
                    index -= self.__num_samples
                image_name = self.__samples[index]['image_name']
                images[num, :, :, :] = self.__image_read(image_name)
                labels[num, :, :, :, :] = self.__samples[index]['yolo_target']
                num += 1
            self.__count += 1
            return images, labels
        else:
            self.__count = 0
            np.random.shuffle(self.__samples)
            raise StopIteration

    def __image_read(self, image_name):
        """
        resize -> RGB转换 -> normalize
        :param image_name: 要读取图像的全路径名称
        :return: image_name对应的图像
        """
        image = cv2.imread(image_name)
        image = cv2.resize(image, (self.__image_size, self.__image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0
        return image


