# coding: utf-8

from yolo import Yolo
import config as cfg
from pascalvoc import PascalVoc
import tensorflow as tf
import numpy as np
import os
import argparse

class YoloTrain(object):
    def __init__(self):
        self.__batch_size = cfg.BATCH_SIZE
        self.__image_size = cfg.IMAGE_SIZE
        self.__cell_size = cfg.CELL_SIZE
        self.__box_per_cell = cfg.BOX_PRE_CELL
        self.__num_class = len(cfg.CLASSES)
        self.__learn_rate_base = cfg.LEARN_RATE_BASE
        self.__max_periods = cfg.MAX_PERIODS
        self.__model_dir = cfg.MODEL_DIR
        self.__model_file = cfg.MODEL_FILE
        self.__log_dir = cfg.LOG_DIR
        self.__moving_ave_decay = cfg.MOVING_AVE_DECAY
        self.__save_iter = cfg.SAVE_ITER

        self.__train_data = PascalVoc('train')
        self.__test_data = PascalVoc('test')

        with tf.name_scope('input'):
            self.__samples = tf.placeholder(dtype=tf.float32, name='samples',
                                            shape=(self.__batch_size, self.__image_size, self.__image_size, 3))
            self.__labels = tf.placeholder(dtype=tf.float32, name='labels',
                                           shape=(self.__batch_size, self.__cell_size, self.__cell_size,
                                                  self.__box_per_cell, 5 + self.__num_class))
            self.__is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        self.__yolo = Yolo()
        self.__yolo_output = self.__yolo.build_network(self.__samples, self.__is_training)
        self.__yolo_loss = self.__yolo.loss(self.__yolo_output, self.__labels)

        with tf.name_scope('learn'):
            self.__learn_rate = tf.Variable(self.__learn_rate_base, trainable=False, name='learn_rate_base')
            moving_ave = tf.train.ExponentialMovingAverage(self.__moving_ave_decay).apply(tf.trainable_variables())
            optimize = tf.train.AdamOptimizer(self.__learn_rate).minimize(self.__yolo_loss)
            with tf.control_dependencies([optimize]):
                with tf.control_dependencies([moving_ave]):
                    self.__train_op = tf.no_op()

        with tf.name_scope('load'):
            self.__load = tf.train.Saver(tf.trainable_variables())
            self.__save = tf.train.Saver(tf.global_variables(), max_to_keep=50)

        with tf.name_scope('summary'):
            tf.summary.scalar('batch_loss', self.__yolo_loss)
            self.__summary_op = tf.summary.merge_all()
            self.__summary_writer = tf.summary.FileWriter(self.__log_dir)
            self.__summary_writer.add_graph(tf.get_default_graph())

        self.__sess = tf.Session()

    def train(self):
        self.__sess.run(tf.global_variables_initializer())
        ckpt_path = os.path.join(self.__model_dir, self.__model_file)
        print 'Restoring weights from:\t %s' % ckpt_path
        self.__load.restore(self.__sess, ckpt_path)

        for period in range(self.__max_periods):
            if period in [20, 50, 80]:
                learning_rate_value = self.__sess.run(
                    tf.assign(self.__learn_rate, self.__sess.run(self.__learn_rate) / 10.0)
                )
                print 'The value of learn rate is:\t%f' % learning_rate_value

            for step, (batch_sample, batch_label) in enumerate(self.__train_data):
                _, summary_value, yolo_loss_value = self.__sess.run(
                    [self.__train_op, self.__summary_op, self.__yolo_loss], feed_dict={
                    self.__samples: batch_sample,
                    self.__labels: batch_label,
                    self.__is_training: True
                })
                if np.isnan(yolo_loss_value):
                    raise ArithmeticError('The gradient is exploded')
                if step % 10:
                    continue
                self.__summary_writer.add_summary(summary_value, period * len(self.__train_data) + step)
                print 'Period:\t%d\tstep:\t%d\ttrain loss:\t%.4f' % (period, step, yolo_loss_value)

            if period % self.__save_iter:
                continue

            total_test_loss = 0.0
            for batch_sample, batch_label in self.__test_data:
                yolo_loss_value = self.__sess.run(self.__yolo_loss, feed_dict={
                    self.__samples: batch_sample,
                    self.__labels: batch_label,
                    self.__is_training: False
                })
                total_test_loss += yolo_loss_value
            test_loss = total_test_loss / len(self.__test_data)
            print 'Period:\t%d\ttest loss:\t%.4f' % (period, test_loss)
            saved_model_name = os.path.join(self.__model_dir, 'yolo.ckpt-%d-%.4f' % (period, test_loss))
            self.__save.save(self.__sess, saved_model_name)
            print 'Saved model:\t%s' % saved_model_name
        self.__summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', default='yolo_coco_initial.ckpt', type=str)
    parser.add_argument('--gpu', default='0, 1', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    if args.model_file is not None:
        cfg.MODEL_FILE = args.model_file
    YoloTrain().train()

