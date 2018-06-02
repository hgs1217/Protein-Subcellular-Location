# @Author:      HgS_1217_
# @Create Date: 2018/5/27

"""
VGG-16 train for classification
"""

import random
import time
import numpy as np
import tensorflow as tf

from config import CKPT_PATH, LOG_PATH, DATASET_PATH
from train.utils import per_class_acc, variable_with_weight_decay, add_loss_summaries, \
    conv_layer, avg_pool_layer, fc_layer, weighted_loss
from data_process.image_preprocessor import ImagePreprocessor


class CNN:
    def __init__(self, raws, labels, test_raws, test_labels, keep_pb=0.5, batch_size=240, epoch_size=100,
                 learning_rate=0.001, start_step=0, loss_array=None):
        """
        Convolutional neural network
        :param raws: list
                    Raw data in training set.
        :param labels: list
                    Labels in training set.
        :param test_raws: list
                    Raw data in test set.
        :param test_labels: list
                    Labels in test set.
        :param keep_pb: float
                    The keep probabilities used in dropout layers.
        :param batch_size: int
                    Batch size.
        :param epoch_size: int
                    Epoch size
        :param learning_rate: float
                    Learning rate
        :param start_step: int
                    Current start step, which is used in further training based on existed model to prevent
                    the disorder of current epoch.
        :param loss_array: list
                    Weight loss array used in loss function.
        """
        self._raws = raws
        self._labels = labels
        self._test_raws = test_raws
        self._test_labels = test_labels
        self._keep_pb = keep_pb
        self._batch_size = batch_size
        self._epoch_size = epoch_size
        self._start_step = start_step
        self._learning_rate = learning_rate
        self._image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
        [_, self._input_width, self._input_height, self._input_channels] = raws[0].shape
        [_, self._label_nums, self._classes] = labels[0].shape
        self._loss_array = [[0.5, 0.5] for _ in range(self._labels)] if loss_array is None else loss_array

        self._x = tf.placeholder(tf.float32, shape=[None, self._input_width, self._input_height, self._input_channels],
                                 name="input_x")
        self._y = tf.placeholder(tf.float32, shape=[None, self._label_nums, self._classes], name="input_y")
        self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._global_step = tf.Variable(0, trainable=False)

    def _build_network_vgg16(self, x, y, is_training):
        """
            VGG16 network construction
            :param x: tensor
                    Raw data
            :param y: tensor
                    Labels given
            :param is_training: boolean
                    Whether it is in the training step.
            :return:
        """
        x_resh = tf.reshape(x, [-1, self._input_width, self._input_height, self._input_channels])
        conv1_1 = conv_layer(x_resh, 3, 1, 128, is_training, name="conv1_1")
        conv1_2 = conv_layer(conv1_1, 3, 1, 128, is_training, name="conv1_2")
        pool1 = avg_pool_layer(conv1_2, 2, 2, name="pool1")

        conv2_1 = conv_layer(pool1, 3, 1, 256, is_training, name="conv2_1")
        conv2_2 = conv_layer(conv2_1, 3, 1, 256, is_training, name="conv2_2")
        pool2 = avg_pool_layer(conv2_2, 2, 2, name="pool2")

        conv3_1 = conv_layer(pool2, 3, 1, 512, is_training, name="conv3_1")
        conv3_2 = conv_layer(conv3_1, 3, 1, 512, is_training, name="conv3_2")
        conv3_3 = conv_layer(conv3_2, 3, 1, 512, is_training, name="conv3_3")
        pool3 = avg_pool_layer(conv3_3, 2, 2, name="pool3")

        fc_in = tf.reshape(pool3, [-1, 3 * 3 * 512])
        fc4 = fc_layer(fc_in, 4096, is_training, name="fc4", relu_flag=True)
        dropout4 = tf.nn.dropout(fc4, self._keep_prob)

        fc5 = fc_layer(dropout4, 4096, is_training, name="fc5", relu_flag=True)
        dropout5 = tf.nn.dropout(fc5, self._keep_prob)

        fc6 = fc_layer(dropout5, self._classes, is_training, name="fc6", relu_flag=False)

        out = fc6
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
        accu = tf.reduce_mean(tf.cast(tf.equal(tf.round((tf.nn.sigmoid(out) - y) * 1.01), self._zeros), tf.float32))
        return loss, out, accu

    def _build_network_lenet(self, x, y, is_training):
        """
            Lenet network construction
            :param x: tensor
                    Raw data
            :param y: tensor
                    Labels given
            :param is_training: boolean
                    Whether it is in the training step.
            :return:
        """
        x_resh = tf.reshape(x, [-1, self._input_width, self._input_height, self._input_channels])
        outfc = []
        for i in range(self._label_nums):   # We train the six labels at the same time
            label_name = "label%d_" % i
            conv1 = conv_layer(x_resh, 5, 1, 32, is_training, name=label_name+"conv1")
            pool1 = avg_pool_layer(conv1, 2, 2, name=label_name+"pool1")

            conv2 = conv_layer(pool1, 5, 1, 64, is_training, name=label_name+"conv2")
            pool2 = avg_pool_layer(conv2, 2, 2, name=label_name+"pool2")

            fc_in = tf.reshape(pool2, [-1, 5 * 5 * 64])
            fc3 = fc_layer(fc_in, 1024, is_training, name=label_name+"fc3", relu_flag=True)
            dropout3 = tf.nn.dropout(fc3, self._keep_prob)

            fc4 = fc_layer(dropout3, self._classes, is_training, name=label_name+"fc4", relu_flag=True)
            outfc.append(fc4)

        out = tf.reshape(tf.concat(outfc, axis=1), [-1, self._label_nums, self._classes])
        loss = weighted_loss(out, y, self._classes, self._loss_array)
        accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 2), tf.argmax(y, 2)), tf.float32), axis=0)
        return loss, out, accu

    def _train_set(self, total_loss, global_step):
        """
            Training operation settings, including optimizer and so on.
            :param total_loss: list, [label number]
            :param global_step:
            :return: train_op
        """
        train_op = [0 for _ in range(self._label_nums)]
        for i in range(self._label_nums):   # We train the six labels at the same time
            loss_averages_op = add_loss_summaries(total_loss[i])

            with tf.control_dependencies([loss_averages_op]):
                opt = tf.train.AdamOptimizer(self._learning_rate)
                grads = opt.compute_gradients(total_loss[i])

            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                train_op[i] = tf.no_op(name='train')

        return train_op

    def _print_class_accu(self, epoch_loss, epoch_accu):
        """
            Print class accuracy
        """
        for i in range(self._label_nums):
            print("\tlabel class %d, loss %g accu %g" % (i, epoch_loss[i], epoch_accu[i]))

    def train(self):
        """
            The main training process.
            :return:
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            loss, prediction, accu = self._build_network_lenet(self._x, self._y, self._is_training)
            train_op = self._train_set(loss, self._global_step)

            saver = tf.train.Saver()
            tf.add_to_collection('prediction', prediction)

            summary_op = tf.summary.merge_all()

            if self._start_step > 0:
                saver.restore(sess, CKPT_PATH)
            else:
                sess.run(tf.global_variables_initializer())

            # summary_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
            loss_summary = tf.summary.scalar("Average_loss", tf.reduce_mean(loss))
            accu_summary = tf.summary.scalar("Prediction_accuracy", tf.reduce_mean(accu))

            for step in range(self._start_step + 1, self._start_step + self._epoch_size + 1):
                print("Training epoch %d/%d" % (step, self._epoch_size))
                total_batch = len(self._raws)
                epoch_loss = np.zeros((total_batch, self._label_nums))
                epoch_accu = np.zeros((total_batch, self._label_nums))
                epoch_total_accu = np.zeros(total_batch)

                for bat in range(total_batch):
                    batch_xs = self._raws[bat]
                    batch_ys = self._labels[bat]
                    _, sum_str, pd, epoch_loss[bat, :], epoch_accu[bat, :], loss_str, accu_str = sess.run(
                        [train_op, summary_op, prediction, loss, accu, loss_summary, accu_summary],
                        feed_dict={self._x: batch_xs, self._y: batch_ys, self._keep_prob: self._keep_pb,
                                   self._is_training: True})
                    epoch_total_accu[bat] = np.mean(np.prod(epoch_accu[bat], axis=0))
                    print("Training epoch %d/%d, batch %d/%d, loss %g, accuracy %g" %
                          (step, self._epoch_size, bat + 1, total_batch, np.mean(epoch_loss[bat]),
                           epoch_total_accu[bat]))
                    if bat % 10 == 9:
                        self._print_class_accu(epoch_loss[bat], epoch_accu[bat])

                print("Training epoch %d/%d finished, loss %g, accuracy %g" %
                      (step, self._epoch_size, np.mean(epoch_loss), np.mean(epoch_total_accu)))
                self._print_class_accu(np.mean(epoch_loss, axis=0), np.mean(epoch_accu, axis=0))
                print("==============================================================")

                # summary_writer.add_summary(sum_str, step)
                # summary_writer.add_summary(loss_str, step)
                # summary_writer.add_summary(accu_str, step)

                if step % 1 == 0:
                    print("Testing epoch {0}".format(step))
                    test_batch = len(self._test_raws)
                    epoch_loss = np.zeros((test_batch, self._label_nums))
                    epoch_accu = np.zeros((test_batch, self._label_nums))
                    epoch_total_accu = np.zeros(total_batch)

                    for bat in range(test_batch):
                        batch_xs = self._test_raws[bat]
                        batch_ys = self._test_labels[bat]
                        pd, epoch_loss[bat, :], epoch_accu[bat, :], loss_str, accu_str = sess.run(
                            [prediction, loss, accu, loss_summary, accu_summary],
                            feed_dict={self._x: batch_xs, self._y: batch_ys, self._keep_prob: 1.0,
                                       self._is_training: False})
                        epoch_total_accu[bat] = np.mean(np.prod(epoch_accu[bat], axis=0))
                        print("Testing epoch %d/%d, batch %d/%d, loss %g, accuracy %g" %
                              (step, self._epoch_size, bat + 1, test_batch, np.mean(epoch_loss[bat]),
                               epoch_total_accu[bat]))
                        if bat % 10 == 9:
                            self._print_class_accu(epoch_loss[bat], epoch_accu[bat])

                    print("Testing epoch %d/%d finished, loss %g, accuracy %g" %
                          (step, self._epoch_size, np.mean(epoch_loss), np.mean(epoch_total_accu)))
                    self._print_class_accu(np.mean(epoch_loss, axis=0), np.mean(epoch_accu, axis=0))
                    print("==============================================================")

                print("saving model.....")
                # saver.save(sess, CKPT_PATH)
                time.sleep(10)
                print("end saving....\n")
