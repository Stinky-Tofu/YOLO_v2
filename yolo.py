# coding: utf-8

import config as cfg
import tensorflow as tf
import numpy as np


class YoloV2(object):
    def __init__(self):
        self.__classes = cfg.CLASSES
        self.__num_class = len(self.__classes)
        self.__box_per_cell = cfg.BOX_PRE_CELL
        self.__batch_size = cfg.BATCH_SIZE
        self.__image_size = cfg.IMAGE_SIZE
        self.__cell_size = cfg.CELL_SIZE
        self.__down_sample_size = 1.0 * self.__image_size / self.__cell_size
        self.__anchor = cfg.ANCHOR

        self.__obj_conf = 1.0
        self.__noobj_conf = 0.5
        self.__obj_coor = 5.0
        self.__obj_prob = 1.0

    def __batch_normalization(self, input_data, is_training, decay=0.95):
        """
        :param input_data: 输入数据，shape为'NHWC'
        :param is_training: 是否在训练，即bn会根据该参数选择mean and variance
        :param decay: 均值方差滑动参数
        :return: BN后的数据
        """
        with tf.variable_scope('BN'):
            input_depth = input_data.get_shape()[-1]
            moving_mean = tf.get_variable(name='moving_mean', shape=input_depth, dtype=tf.float32,
                                          initializer=tf.zeros_initializer, trainable=False)
            moving_var = tf.get_variable(name='moving_var', shape=input_depth, dtype=tf.float32,
                                         initializer=tf.ones_initializer, trainable=False)

            def mean_and_var_update():
                axes = range(len(input_data.get_shape()) - 1)
                batch_mean = tf.reduce_mean(input_data, axis=axes)
                batch_var = tf.reduce_mean(tf.pow(input_data - batch_mean, 2), axis=axes)
                with tf.control_dependencies([tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay)),
                                              tf.assign(moving_var, moving_var * decay + batch_var * (1 - decay))]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, variance = tf.cond(is_training, mean_and_var_update, lambda:(moving_mean, moving_var))
            shift = tf.get_variable(name='shift', shape=input_depth, dtype=tf.float32,
                                    initializer=tf.zeros_initializer, trainable=True)
            scale = tf.get_variable(name='scale', shape=input_depth, dtype=tf.float32,
                                    initializer=tf.ones_initializer, trainable=True)
            return tf.nn.batch_normalization(input_data, mean, variance, shift, scale, 1e-8)

    def __conv(self, var_scope, input_data, filter_size, is_training,
               stride=(1, 1, 1, 1), padding='SAME', activation=True, BN=True):
        """
        :param var_scope: conv层的名称空间
        :param input_data: conv层的输入数据，格式为'NHWC'
        :param filter_size: filter的size，格式为[filter_height, filter_width, in_channels, out_channels]
        :param stride: filter的stride，格式为[1, stride, stride, 1]
        :param padding: 是否padding，'SAME' or 'VALID'
        :param activation: 是否添加激活函数，默认使用relu激活函数
        :param BN: 是否BN，True or False
        :param is_training: 是否在训练，True or False
        :return: 卷积后的数据
        """
        with tf.variable_scope(var_scope):
            weight = tf.get_variable(name='weight', shape=filter_size, dtype=tf.float32, trainable=True,
                                      initializer=tf.random_normal_initializer(stddev=0.01))
            bias = tf.get_variable(name='bias', shape=filter_size[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input=input_data, filter=weight, strides=stride, padding=padding)
            if BN:
                conv = self.__batch_normalization(input_data=conv, is_training=is_training)
            if activation:
                conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
        return conv

    def __pool(self, var_scope, input_data, ksize=(1,2,2,1), stride=(1,2,2,1),
               padding='SAME', pooling=tf.nn.max_pool):
        """
        :param var_scope: pooling层的命名空间
        :param input_data: pooling层的输入数据，格式为'NHWC'
        :param ksize: pooling的size，格式为[1, pooling_height, pooling_width, 1]
        :param stride: pooling的stride，格式为[1, stride, stride, 1]
        :param padding: 是否padding 'SAME' or 'VALID'
        :param pooling: 选择用哪个pooling层
        :return: 池化后的数据
        """
        with tf.variable_scope(var_scope):
            pool_output = pooling(value=input_data, ksize=ksize, strides=stride, padding=padding)
        return pool_output

    def __reorg(self, input_data):
        """
        :param input_data: 输入数据shape为(batch_size, height, width, channel)
        :return: 返回数据的shape为(batch_size, height/2, width/2, channel*4)
        """
        with tf.name_scope('reorg'):
            output1 = input_data[:, ::2, ::2, :]
            output2 = input_data[:, ::2, 1::2, :]
            output3 = input_data[:, 1::2, ::2, :]
            output4 = input_data[:, 1::2, 1::2, :]
            output_data = tf.concat([output1, output2, output3, output4], axis=-1)
        return output_data

    def build_network(self, input_data, is_training):
        """
        :param input_data: 输入数据，格式为'NHWC'
        :param is_training: 是否在训练，需要是类型为bool的tensorflow 变量，不能使用python的bool类型
        :return: yolo的输出数据，shape为(batch_size, cell_size, cell_size, box_per_cell * (5+num_classes))
        """
        with tf.variable_scope('YOLO'):
            net = self.__conv('conv_0', input_data, [3, 3, 3, 32], is_training)
            net = self.__pool('pool_0', net)

            net = self.__conv('conv_1', net, [3, 3, 32, 64], is_training)
            net = self.__pool('pool_1', net)

            net = self.__conv('conv_2', net, [3, 3, 64, 128], is_training)
            net = self.__conv('conv_3', net, [1, 1, 128, 64], is_training)
            net = self.__conv('conv_4', net, [3, 3, 64, 128],is_training)
            net = self.__pool('pool_2', input_data=net)

            net = self.__conv('conv_5', net, [3, 3, 128, 256], is_training)
            net = self.__conv('conv_6', net, [1, 1, 256, 128], is_training)
            net = self.__conv('conv_7', net, [3, 3, 128, 256], is_training)
            net = self.__pool('pool_3', net)

            net = self.__conv('conv_8', net, [3, 3, 256, 512], is_training)
            net = self.__conv('conv_9', net, [1, 1, 512, 256], is_training)
            net = self.__conv('conv_10', net, [3, 3, 256, 512], is_training)
            net = self.__conv('conv_11', net, [1, 1, 512, 256], is_training)
            net17 = self.__conv('conv_12', net, [3, 3, 256, 512], is_training)
            net = self.__pool('pool_4', net17)

            net = self.__conv('conv_13', net, [3, 3, 512, 1024], is_training)
            net = self.__conv('conv_14', net, [1, 1, 1024, 512], is_training)
            net = self.__conv('conv_15', net, [3, 3, 512, 1024], is_training)
            net = self.__conv('conv_16', net, [1, 1, 1024, 512], is_training)
            net = self.__conv('conv_17', net, [3, 3, 512, 1024], is_training)

            net = self.__conv('conv_18', net, [3, 3, 1024, 1024], is_training)
            net25 = self.__conv('conv_19', net, [3, 3, 1024, 1024], is_training)

            shortcut = self.__conv('shortcut', net17, [1, 1, 512, 64], is_training)
            shortcut = self.__reorg(shortcut)
            net = tf.concat([net25, shortcut], axis=3)

            net = self.__conv('conv_20', net, [3, 3, net.get_shape()[-1], 1024], is_training)
            net = self.__conv('conv_21', net, [1, 1, 1024, self.__box_per_cell * (5 + self.__num_class)],
                              activation=False, BN=False, is_training=is_training)
            return net

    def loss(self, predict, label):
        """
        :param predict: predict是yolo的输出，shape为(batch_size, cell_size, cell_size, box_per_cell * (5 + num_class))
        :param label: label是标签，shape为(batch_size, cell_size, cell_size, box_per_cell, 5 + num_class)
        :return: YOLO的loss
        """
        with tf.name_scope('yolo_loss'):
            predict = tf.reshape(predict, (self.__batch_size, self.__cell_size, self.__cell_size,
                                           self.__box_per_cell, self.__num_class + 5))
            pred_conf = predict[:, :, :, :, 0:1]
            pred_coor = predict[:, :, :, :, 1:5]
            pred_prob = predict[:, :, :, :, 5:]

            # 1、将yolo的输出转换成用于计算loss的形式
            # coor存储的是(dx,dy,dw,dh)，(dx,dy,dw,dh)的大小都是相对于down_sample_size的
            # (down_sample_size = image_size / cell_size)
            # dx,dy是预测框的中心坐标相对于其所在grid的左上角坐标的偏移量、dw,dh是预测框的宽高相对于Anchor宽高的缩放量
            pred_conf = tf.sigmoid(pred_conf)
            pred_coor = tf.concat([tf.sigmoid(pred_coor[:, :, :, :, 0:2]),
                                   tf.exp(pred_coor[:, :, :, :, 2:])], axis=-1)
            pred_prob = tf.nn.softmax(pred_prob)

            # 2、将标签转换成用于计算loss的形式
            # (1)计算所有Anchor与相应GT的IOU
            GT_coor = label[:, :, :, :, 1:5]
            y, x = np.mgrid[:self.__cell_size, :self.__cell_size]
            grid_coor = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]], axis=-1)
            grid_coor = np.repeat(grid_coor[:, :, np.newaxis, :], self.__box_per_cell, axis=-2)
            grid_coor = np.repeat(grid_coor[np.newaxis, :, :, :, :], self.__batch_size, axis=0)

            anchor_coor = np.zeros((self.__batch_size, self.__cell_size, self.__cell_size, self.__box_per_cell, 4))
            anchor_coor[:, :, :, :, :2] = grid_coor + 0.5
            anchor_coor[:, :, :, :, 2:] = [[w, h] for w, h in self.__anchor]
            anchor_coor = tf.constant(anchor_coor, dtype=tf.float32, name='Anchors')
            iou = self.__calc_iou(GT_coor, anchor_coor)

            # (2)将label转换成用于计算loss的形式(tx, ty, tw, th)
            # GT_coor存储的是(x,y,w,h)，(x,y,w,h)的大小都是相对于down_sample_size的，其中(x,y)是GT的中心坐标
            # 此处计算出GT的中心坐标相对于其所在grid的左上角坐标的偏移量(tx, ty)、以及GT的宽高相对于Anchor宽高的缩放量(tw, th)
            # GT落入某个grid，那么该grid的所有Anchor全部先打上位置标签、置信度标签、类别标签，其他grid的Anchor的标签为0
            xy_shift = tf.maximum(GT_coor[:, :, :, :, :2] - grid_coor, 0.0)
            wh_scale = 1.0 * GT_coor[:, :, :, :, 2:] / anchor_coor[:, :, :, :, 2:]
            label_coor = tf.concat([xy_shift, wh_scale], axis=-1)
            label_conf = label[:, :, :, :, 0:1]
            label_prob = label[:, :, :, :, 5:]

            # 3、计算loss
            # iou是GT与相应grid中所有Anchor的iou
            # 只有当某个grid中的某个Anchor要负责某个GT时，该Anchor对应的responsible_obj中的位置才为1，否则为0
            max_iou = tf.reduce_max(iou, axis=-1, keep_dims=True)
            # 因为[0, 0, 0...]的最大值为0，所以需要在后面减去tf.to_float(tf.equal(max_iou, 0))，以解决这种情况
            responsible_obj = tf.to_float(tf.equal(max_iou, iou)) - tf.to_float(tf.equal(max_iou, 0))
            responsible_obj = responsible_obj[:, :, :, :, tf.newaxis]
            no_responsible_obj = 1.0 - responsible_obj
            iou_lessthan_thr = tf.to_float(iou < 0.6)
            iou_lessthan_thr = iou_lessthan_thr[:, :, :, :, tf.newaxis]

            loss_coor = self.__obj_coor * responsible_obj * tf.square(pred_coor - label_coor)
            loss_conf = self.__obj_conf * responsible_obj * tf.square(pred_conf - label_conf) + \
                        self.__noobj_conf * no_responsible_obj * iou_lessthan_thr * tf.square(pred_conf)
            loss_prob = self.__obj_prob * responsible_obj * tf.square(pred_prob - label_prob)

            loss = tf.concat([loss_conf, loss_coor, loss_prob], axis=-1)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]), name='yolo_v2_loss')
            return loss

    def __calc_iou(self, boxes1, boxes2):
        """
        :param boxes1: 输入boxes的shape为必须(batch_size, image_height, image_width, box_per_cell, 4)
        存储结构为(x,y,w,h)，其中(x,y)是bbox的中心坐标
        :param boxes2:
        :return: 返回boxes1和boxes2的IOU，IOU的shape为(batch_size, image_height, image_width, box_per_cell)
        """
        boxes1_area = boxes1[:, :, :, :, 2] * boxes1[:, :, :, :, 3]
        boxes2_area = boxes2[:, :, :, :, 2] * boxes2[:, :, :, :, 3]

        # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
        # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
        boxes1 = tf.concat([boxes1[:, :, :, :, :2] - boxes1[:, :, :, :, 2:] * 0.5,
                            boxes1[:, :, :, :, :2] + boxes1[:, :, :, :, 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[:, :, :, :, :2] - boxes2[:, :, :, :, 2:] * 0.5,
                            boxes2[:, :, :, :, :2] + boxes2[:, :, :, :, 2:] * 0.5], axis=-1)

        # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
        left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        right_down = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[:, :, :, :, 0] * inter_section[:, :, :, :, 1]
        union_area = boxes1_area + boxes2_area - inter_area
        IOU = 1.0 * inter_area / union_area
        return IOU









