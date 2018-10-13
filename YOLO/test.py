# coding: utf-8

import numpy as np
import config as cfg
import cv2
import tensorflow as tf
from yolo import Yolo
import os
import random
import colorsys
import argparse


class YoloTest(object):
    def __init__(self):
        self.__image_size = cfg.IMAGE_SIZE
        self.__cell_size = cfg.CELL_SIZE
        self.__down_sample_size = 1.0 * self.__image_size / self.__cell_size
        self.__box_per_cell = cfg.BOX_PRE_CELL
        self.__classes = cfg.CLASSES
        self.__num_class = len(cfg.CLASSES)
        self.__anchor = cfg.ANCHOR
        self.__prob_threshold = cfg.PROB_THRESHOLD
        self.__nms_threshold = cfg.NMS_THRESHOLD

        self.__yolo_input = tf.placeholder(shape=(1, self.__image_size, self.__image_size, 3),
                                           dtype=tf.float32, name='input')
        self.__is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        self.__yolo_output = Yolo().build_network(self.__yolo_input, self.__is_training)
        self.__sess = tf.Session()
        self.__saver = tf.train.Saver(tf.global_variables())
        self.__saver.restore(self.__sess, os.path.join(cfg.MODEL_DIR, cfg.MODEL_FILE))

    def detect(self, image):
        original_image = np.copy(image)
        original_h, original_w, _ = original_image.shape

        yolo_input = self.__pre_process(image)
        yolo_output = self.__sess.run(self.__yolo_output, feed_dict={
            self.__yolo_input: yolo_input,
            self.__is_training: False
        })

        pred_conf, pred_coor, pred_prob = self.__decode(yolo_output)
        bboxes, scores, classes = self.__post_process(pred_conf, pred_coor, pred_prob)
        bboxes[:, [0, 2]] = 1.0 * bboxes[:, [0, 2]] * original_w / self.__image_size
        bboxes[:, [1, 3]] = 1.0 * bboxes[:, [1, 3]] * original_h / self.__image_size
        image = self.__draw_bbox(original_image, bboxes, scores, classes)
        return image

    def __pre_process(self, image):
        """
        RGB转换->将原始图像resize为yolo要求的输入尺寸->normalize->增加batch_size维
        :param image: shape为(original_h, original_w, 3)
        :return: shape为(1, image_size, image_size, 3)
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.__image_size, self.__image_size))
        image = image / 255.0 * 2.0 - 1.0
        image = image[np.newaxis,...]
        return image

    def __sigmoid(self, arr):
        """
        对数组arr中的每个元素执行sigmoid计算
        :param arr: 任意shape的数组
        :return: sigmoid后的数组
        """
        arr = np.array(arr, dtype=np.float128)
        return 1.0 / (1.0 + np.exp(-1.0 * arr))

    def __softmax(self, arr):
        """
        :param arr: arr最后一维必须是logic维
        :return: softmax后的arr
        """
        arr = np.array(arr, dtype=np.float128)
        arr_exp = np.exp(arr)
        return arr_exp / np.expand_dims(np.sum(arr_exp, axis=-1), axis=-1)

    def __decode(self, predict):
        """
        将yolo的输出进行解码，获得每个Anchor的(confidence, xmin, ymin, xmax, ymax, probability)
        其中xmin, ymin, xmax, ymax的大小都是相对于image_size的
        :param predict: yolo的输出，shape为(1, cell_size, cell_size, box_per_cell * (num_class + 5))
        :return:
        pred_conf，shape为(cell_size, cell_size, box_per_cell, 1）
        pred_coor，shape为(cell_size, cell_size, box_per_cell, 4)
        perd_prob，shape为(cell_size, cell_size, box_per_cell, num_class)
        """
        predict = np.reshape(predict, (self.__cell_size, self.__cell_size, self.__box_per_cell, self.__num_class + 5))
        pred_conf = predict[:, :, :, 0:1]
        pred_coor = predict[:, :, :, 1:5]
        pred_prob = predict[:, :, :, 5:]

        # 获取yolo的输出feature map中每个grid左上角的坐标，以及每个Anchor的wh(wh是相对于down_sample_size的)
        # 需注意的是图像的坐标轴方向为
        #  - - - - > x
        # |
        # |
        # ↓
        # y
        # 在图像中标注坐标时通常用(y,x)，但此处为了与coor的存储格式(dx, dy, dw, dh)保持一致，将grid的坐标存储为(x, y)的形式
        y, x = np.mgrid[:self.__cell_size, :self.__cell_size]
        grid_coor = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]], axis=-1)
        grid_coor = np.repeat(grid_coor[:, :, np.newaxis, :], self.__box_per_cell, axis=-2)
        anchor_wh = [[w, h] for w, h in self.__anchor]

        pred_conf = self.__sigmoid(pred_conf)
        pred_coor = np.concatenate([self.__sigmoid(pred_coor[:, :, :, :2]) + grid_coor,
                                    np.exp(pred_coor[:, :, :, 2:]) * anchor_wh], axis=-1)
        pred_coor = np.concatenate([pred_coor[:, :, :, :2] - pred_coor[:, :, :, 2:] * 0.5,
                                    pred_coor[:, :, :, :2] + pred_coor[:, :, :, 2:] * 0.5], axis=-1)
        pred_coor = pred_coor * self.__down_sample_size
        pred_prob = self.__softmax(pred_prob)
        return pred_conf, pred_coor, pred_prob

    def __post_process(self, pred_conf, pred_coor, pred_prob):
        """
        对yolo输出的解码数据进行NMS
        :param pred_conf: shape为(cell_size, cell_size, box_per_cell, 1)
        :param pred_coor: shape为(cell_size, cell_size, box_per_cell, 4)
        :param pred_prob: shape为(cell_size, cell_size, box_per_cell, num_class)
        :return: 假设NMS后剩下N个bbox，那么bboxes的shape为(N, 4)、sores的shape为(N,)、classes的shape为(N,)
        """
        pred_conf = np.reshape(pred_conf, (-1,))
        pred_coor = np.reshape(pred_coor, (-1, 4))
        pred_prob = np.reshape(pred_prob, (-1, self.__num_class))

        # 如果某个bbox超出图像范围，那么将超出部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [self.__image_size-1, self.__image_size-1])], axis=-1)

        # 获取所有bbox的置信度
        classes = np.argmax(pred_prob, -1)
        scores = pred_conf * pred_prob[np.arange(len(pred_prob)), classes]

        # 获取所有置信度大于prob_threshold的bbox，然后根据置信度从这些bbox中取top_k个(默认为400)
        keep_index = scores > self.__prob_threshold
        bboxes = pred_coor[keep_index]
        scores = scores[keep_index]
        classes = classes[keep_index]
        bboxes, scores, classes = self.__bboxes_sort(bboxes, scores, classes)

        # 对这top_k个bbox进行NMS
        bboxes, scores, classes = self.__nms(bboxes, scores, classes)
        return bboxes, scores, classes

    def __bboxes_sort(self, bboxes, scores, classes, top_k=400):
        """
        对scores排序后，取top_k个bbox、以及这些bbox对应的score和class
        :param bboxes: shape为(..., 4)
        :param scores: shape为(...,)
        :param classes: shape为(...,)
        :param top_k:
        :return: bboxes的shape为(top_k, 4)、scores的shape为(top_k,)、classes的shape为(top_k,)
        """
        index = np.argsort(-scores)
        bboxes = bboxes[index][:top_k]
        scores = scores[index][:top_k]
        classes = classes[index][:top_k]
        return bboxes, scores, classes

    def __calc_iou(self, boxes1, boxes2):
        """
        :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
        :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
        :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
        """
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        # 计算出boxes1和boxes2相交部分的宽、高
        # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        IOU = 1.0 * inter_area / union_area
        return IOU

    def __nms(self, bboxes, scores, classes):
        """
        :param bboxes: shape为(num_boxes, 4)，坐标存储格式需要是(xmin, ymin, xmax, ymax)
        :param scores: shape为(num_boxes,)
        :param classes: shape为(num_boxes,)
        :return: 假设NMS后剩下N个bbox，那么bboxes的shape为(N, 4)、scores的shape为(N,)、classes的shape为(N,)
        """
        keep_bboxes = np.ones(scores.shape, dtype=np.bool)
        for i in range(scores.size - 1):
            if keep_bboxes[i]:
                iou = self.__calc_iou(bboxes[i], bboxes[(i + 1):])
                keep_overlap = np.logical_or(iou < self.__nms_threshold, classes[(i + 1):] != classes[i])
                keep_bboxes[(i + 1):] = np.logical_and(keep_bboxes[(i + 1):], keep_overlap)
        idxes = np.where(keep_bboxes)
        return bboxes[idxes], scores[idxes], classes[idxes],

    def __draw_bbox(self, original_image, bboxes, scores, classes):
        """
        :param original_image: 检测的原始图片，shape为(original_image_h, original_image_w, 3)
        :param bboxes: 预测框，shape为(..., 4)，坐标存储格式为(xmin, ymin, xmax, ymax)，这些坐标都是相对于original image的
        :param scores: 预测框的confidence，shape为(...,)
        :param classes: 预测框的类别，shape为(...,)
        :return:
        """
        hsv_tuples = [(1.0 * x / self.__num_class, 1., 1.) for x in range(self.__num_class)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        image_h, image_w, _ = original_image.shape
        for i, bbox in enumerate(bboxes):
            bbox_color = colors[classes[i]]
            bbox_thick = int(float(image_h + image_w) / 600)
            cv2.rectangle(original_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, bbox_thick)

            bbox_mess = '%s: %.3f' % (self.__classes[classes[i]], scores[i])
            text_loc = (int(bbox[0]), int(bbox[1] + 5) if bbox[1] < 20 else int(bbox[1] - 5))
            cv2.putText(original_image, bbox_mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image_h, (255, 255, 255), bbox_thick / 3)
        return original_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', default='yolo.ckpt', type=str)
    parser.add_argument('--image_path', default='./data/image.jpg', type=str)
    parser.add_argument('--image_save_path', default='./data/image_detected.jpg', type=str)
    parser.add_argument('--show', default='False', type=str)
    parser.add_argument('--gpu', default='0,1', type=str)
    args = parser.parse_args()
    model_file = args.model_file
    image_path = args.image_path
    image_save_path = args.image_save_path
    show = args.show
    gpu = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    cfg.MODEL_FILE = model_file
    model_path = os.path.join(cfg.MODEL_DIR, model_file) + '.index'
    if not os.path.exists(model_path):
        raise RuntimeError(str('You must enter a valid model file in directory: %s' % cfg.MODEL_DIR))
    if not os.path.exists(image_path):
        raise RuntimeError('You must enter a valid path of image')

    T = YoloTest()
    image = cv2.imread(image_path)
    image = T.detect(image)
    cv2.imwrite(image_save_path, image)
    if show == 'True':
        cv2.imshow('image', image)
        cv2.waitKey(0)
