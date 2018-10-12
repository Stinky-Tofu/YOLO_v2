# coding: utf-8

import tensorflow as tf
import os
import shutil
import config as cfg
from yolo import YoloV2


name_dict = {
        'weight': 'YOLO/conv_0/weight', 'biases': 'YOLO/conv_0/bias',
        'scale': 'YOLO/conv_0/BN/scale', 'shift': 'YOLO/conv_0/BN/shift',
        'save_mean': 'YOLO/conv_0/BN/moving_mean', 'save_variance': 'YOLO/conv_0/BN/moving_var',

        'weight_1': 'YOLO/conv_1/weight', 'biases_1': 'YOLO/conv_1/bias',
        'scale_1': 'YOLO/conv_1/BN/scale', 'shift_1': 'YOLO/conv_1/BN/shift',
        'save_mean_1': 'YOLO/conv_1/BN/moving_mean', 'save_variance_1': 'YOLO/conv_1/BN/moving_var',

        'weight_2': 'YOLO/conv_2/weight', 'biases_2': 'YOLO/conv_2/bias',
        'scale_2': 'YOLO/conv_2/BN/scale', 'shift_2': 'YOLO/conv_2/BN/shift',
        'save_mean_2': 'YOLO/conv_2/BN/moving_mean', 'save_variance_2': 'YOLO/conv_2/BN/moving_var',

        'weight_3': 'YOLO/conv_3/weight', 'biases_3': 'YOLO/conv_3/bias',
        'scale_3': 'YOLO/conv_3/BN/scale', 'shift_3': 'YOLO/conv_3/BN/shift',
        'save_mean_3': 'YOLO/conv_3/BN/moving_mean', 'save_variance_3': 'YOLO/conv_3/BN/moving_var',

        'weight_4': 'YOLO/conv_4/weight', 'biases_4': 'YOLO/conv_4/bias',
        'scale_4': 'YOLO/conv_4/BN/scale', 'shift_4': 'YOLO/conv_4/BN/shift',
        'save_mean_4': 'YOLO/conv_4/BN/moving_mean', 'save_variance_4': 'YOLO/conv_4/BN/moving_var',

        'weight_5': 'YOLO/conv_5/weight', 'biases_5': 'YOLO/conv_5/bias',
        'scale_5': 'YOLO/conv_5/BN/scale', 'shift_5': 'YOLO/conv_5/BN/shift',
        'save_mean_5': 'YOLO/conv_5/BN/moving_mean', 'save_variance_5': 'YOLO/conv_5/BN/moving_var',

        'weight_6': 'YOLO/conv_6/weight', 'biases_6': 'YOLO/conv_6/bias',
        'scale_6': 'YOLO/conv_6/BN/scale', 'shift_6': 'YOLO/conv_6/BN/shift',
        'save_mean_6': 'YOLO/conv_6/BN/moving_mean', 'save_variance_6': 'YOLO/conv_6/BN/moving_var',

        'weight_7': 'YOLO/conv_7/weight', 'biases_7': 'YOLO/conv_7/bias',
        'scale_7': 'YOLO/conv_7/BN/scale', 'shift_7': 'YOLO/conv_7/BN/shift',
        'save_mean_7': 'YOLO/conv_7/BN/moving_mean', 'save_variance_7': 'YOLO/conv_7/BN/moving_var',

        'weight_8': 'YOLO/conv_8/weight', 'biases_8': 'YOLO/conv_8/bias',
        'scale_8': 'YOLO/conv_8/BN/scale', 'shift_8': 'YOLO/conv_8/BN/shift',
        'save_mean_8': 'YOLO/conv_8/BN/moving_mean', 'save_variance_8': 'YOLO/conv_8/BN/moving_var',

        'weight_9': 'YOLO/conv_9/weight', 'biases_9': 'YOLO/conv_9/bias',
        'scale_9': 'YOLO/conv_9/BN/scale', 'shift_9': 'YOLO/conv_9/BN/shift',
        'save_mean_9': 'YOLO/conv_9/BN/moving_mean', 'save_variance_9': 'YOLO/conv_9/BN/moving_var',

        'weight_10': 'YOLO/conv_10/weight', 'biases_10': 'YOLO/conv_10/bias',
        'scale_10': 'YOLO/conv_10/BN/scale', 'shift_10': 'YOLO/conv_10/BN/shift',
        'save_mean_10': 'YOLO/conv_10/BN/moving_mean', 'save_variance_10': 'YOLO/conv_10/BN/moving_var',

        'weight_11': 'YOLO/conv_11/weight', 'biases_11': 'YOLO/conv_11/bias',
        'scale_11': 'YOLO/conv_11/BN/scale', 'shift_11': 'YOLO/conv_11/BN/shift',
        'save_mean_11': 'YOLO/conv_11/BN/moving_mean', 'save_variance_11': 'YOLO/conv_11/BN/moving_var',

        'weight_12': 'YOLO/conv_12/weight', 'biases_12': 'YOLO/conv_12/bias',
        'scale_12': 'YOLO/conv_12/BN/scale', 'shift_12': 'YOLO/conv_12/BN/shift',
        'save_mean_12': 'YOLO/conv_12/BN/moving_mean', 'save_variance_12': 'YOLO/conv_12/BN/moving_var',

        'weight_13': 'YOLO/conv_13/weight', 'biases_13': 'YOLO/conv_13/bias',
        'scale_13': 'YOLO/conv_13/BN/scale', 'shift_13': 'YOLO/conv_13/BN/shift',
        'save_mean_13': 'YOLO/conv_13/BN/moving_mean', 'save_variance_13': 'YOLO/conv_13/BN/moving_var',

        'weight_14': 'YOLO/conv_14/weight', 'biases_14': 'YOLO/conv_14/bias',
        'scale_14': 'YOLO/conv_14/BN/scale', 'shift_14': 'YOLO/conv_14/BN/shift',
        'save_mean_14': 'YOLO/conv_14/BN/moving_mean', 'save_variance_14': 'YOLO/conv_14/BN/moving_var',

        'weight_15': 'YOLO/conv_15/weight', 'biases_15': 'YOLO/conv_15/bias',
        'scale_15': 'YOLO/conv_15/BN/scale', 'shift_15': 'YOLO/conv_15/BN/shift',
        'save_mean_15': 'YOLO/conv_15/BN/moving_mean', 'save_variance_15': 'YOLO/conv_15/BN/moving_var',

        'weight_16': 'YOLO/conv_16/weight', 'biases_16': 'YOLO/conv_16/bias',
        'scale_16': 'YOLO/conv_16/BN/scale', 'shift_16': 'YOLO/conv_16/BN/shift',
        'save_mean_16': 'YOLO/conv_16/BN/moving_mean', 'save_variance_16': 'YOLO/conv_16/BN/moving_var',

        'weight_17': 'YOLO/conv_17/weight', 'biases_17': 'YOLO/conv_17/bias',
        'scale_17': 'YOLO/conv_17/BN/scale', 'shift_17': 'YOLO/conv_17/BN/shift',
        'save_mean_17': 'YOLO/conv_17/BN/moving_mean', 'save_variance_17': 'YOLO/conv_17/BN/moving_var',

        'weight_18': 'YOLO/conv_18/weight', 'biases_18': 'YOLO/conv_18/bias',
        'scale_18': 'YOLO/conv_18/BN/scale', 'shift_18': 'YOLO/conv_18/BN/shift',
        'save_mean_18': 'YOLO/conv_18/BN/moving_mean', 'save_variance_18': 'YOLO/conv_18/BN/moving_var',

        'weight_19': 'YOLO/conv_19/weight', 'biases_19': 'YOLO/conv_19/bias',
        'scale_19': 'YOLO/conv_19/BN/scale', 'shift_19': 'YOLO/conv_19/BN/shift',
        'save_mean_19': 'YOLO/conv_19/BN/moving_mean', 'save_variance_19': 'YOLO/conv_19/BN/moving_var',

        'weight_20': 'YOLO/shortcut/weight', 'biases_20': 'YOLO/shortcut/bias',
        'scale_20': 'YOLO/shortcut/BN/scale', 'shift_20': 'YOLO/shortcut/BN/shift',
        'save_mean_20': 'YOLO/shortcut/BN/moving_mean', 'save_variance_20': 'YOLO/shortcut/BN/moving_var',

        'weight_21': 'YOLO/conv_20/weight', 'biases_21': 'YOLO/conv_20/bias',
        'scale_21': 'YOLO/conv_20/BN/scale', 'shift_21': 'YOLO/conv_20/BN/shift',
        'save_mean_21': 'YOLO/conv_20/BN/moving_mean', 'save_variance_21': 'YOLO/conv_20/BN/moving_var'
    }

# # 获取yolo_coco中的权重（除了最后一个卷积层的权重，因为coco有80类，而pascal_voc只有20类）
# # 在计算图中定义变量，这些变量的名字是我们想要的名字，初始值是从yolo_coco中获取的权重，然后将计算图保存为yolo_coco_renamed
# if __name__ == '__main__':
#     log_dir = './log/yolo_coco_renamed'
#     for var_name in name_dict:
#         value = tf.contrib.framework.load_variable(os.path.join(cfg.MODEL_DIR, 'yolo_coco.ckpt'), var_name)
#         tf.Variable(value, name=name_dict[var_name])
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         saver.save(sess, os.path.join(cfg.MODEL_DIR, 'yolo_coco_renamed.ckpt'))
#
#         if os.path.exists(log_dir):
#             shutil.rmtree(log_dir)
#         Summary = tf.summary.FileWriter(log_dir)
#         Summary.add_graph(tf.get_default_graph())
#         Summary.close()

# 将修改名字后的yolo_coco_renamed模型（这个模型中没有yolo_v2中的最后一个卷积层的权重）来初始化yolo计算图中的可训练变量
# 只加载可训练变量
if __name__ == '__main__':
    # 定义输入
    with tf.name_scope('input'):
        samples = tf.placeholder(dtype=tf.float32, shape=(cfg.BATCH_SIZE, cfg.IMAGE_SIZE,
                                                          cfg.IMAGE_SIZE, 3), name='samples')
        labels = tf.placeholder(dtype=tf.float32, shape=(cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE,
                                                         cfg.BOX_PRE_CELL, 5 + len(cfg.CLASSES)), name='labels')
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    # 创建YOLO network
    yolo = YoloV2()
    yolo_output = yolo.build_network(samples, is_training)
    yolo_loss = yolo.loss(yolo_output, labels)

    # 加载和保存模型
    with tf.name_scope('load'):
        var_dict = {var.op.name: var for var in tf.global_variables()}
        var_list = [var_dict[name_dict[var_name]] for var_name in name_dict]
        load = tf.train.Saver(var_list)
        save = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt_path = os.path.join(cfg.MODEL_DIR, 'yolo_coco_renamed.ckpt')
        print 'Restoring weights from:\t %s' % ckpt_path
        load.restore(sess, ckpt_path)
        save.save(sess, os.path.join(cfg.MODEL_DIR, 'yolo_coco_initial.ckpt'))
