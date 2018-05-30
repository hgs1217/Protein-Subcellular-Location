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
from train.utils import per_class_acc
from data_process.image_preprocessor import ImagePreprocessor


def variable_with_weight_decay(name, shape, initializer, wd=None):
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def batch_norm_layer(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = variable_with_weight_decay('beta', params_shape, initializer=tf.truncated_normal_initializer())
    gamma = variable_with_weight_decay('gamma', params_shape, initializer=tf.truncated_normal_initializer())

    batch_mean, batch_var = tf.nn.moments(x, axis, name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_training, mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def conv2d(x, w, stride, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, ksize, stride, feature_num, is_training, name=None, padding="SAME", groups=1):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [ksize, ksize, int(x.get_shape()[-1]) / groups, feature_num], dtype="float")
        b = tf.get_variable("b", [feature_num], dtype="float")

        x_split = tf.split(x, groups, 3)
        w_split = tf.split(w, groups, 3)

        feature_map_list = [conv2d(x_, w_, stride, padding) for x_, w_ in zip(x_split, w_split)]
        feature_map = tf.concat(feature_map_list, 3)

        out = tf.nn.bias_add(feature_map, b)
        norm = batch_norm_layer(out, is_training)
        feature_shape = list(map(lambda x: -1 if not x else x, feature_map.get_shape().as_list()))
        return tf.nn.relu(tf.reshape(norm, feature_shape), name=scope.name)


def avg_pool_layer(x, ksize, stride, name=None, padding="SAME"):
    return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)


def fc_layer(x, feature_num, is_training, name=None, relu_flag=True):
    with tf.variable_scope(name) as scope:
        w = variable_with_weight_decay('w', shape=[x.get_shape()[-1], feature_num],
                                       initializer=tf.orthogonal_initializer(), wd=None)
        b = tf.get_variable("b", [feature_num], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(tf.matmul(x, w), b)
        norm = batch_norm_layer(bias, is_training)
        return tf.nn.relu(norm) if relu_flag else norm


def weighted_loss(lgts, lbs, num_classes, loss_array):
    with tf.name_scope('loss'):
        epsilon = tf.constant(value=1e-10)
        labels = lbs
        logits = lgts + epsilon

        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(tf.nn.softmax(logits) + epsilon),
                                                   np.array(loss_array)), axis=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


class VGG16:
    def __init__(self, raws, labels, test_raws, test_labels, keep_pb=0.5, batch_size=240, epoch_size=100,
                 learning_rate=0.001, classes=1, start_step=0, loss_array=None):
        self._raws = raws
        self._labels = labels
        self._test_raws = test_raws
        self._test_labels = test_labels
        self._keep_pb = keep_pb
        self._batch_size = batch_size
        self._epoch_size = epoch_size
        self._start_step = start_step
        self._learning_rate = learning_rate
        self._classes = classes
        self._image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
        [_, self._input_width, self._input_height, self._input_channels] = raws[0].shape
        self._loss_array = [0.5, 0.5] if loss_array is None else loss_array

        self._x = tf.placeholder(tf.float32, shape=[None, self._input_width, self._input_height, self._input_channels],
                                 name="input_x")
        self._y = tf.placeholder(tf.float32, shape=[None, self._classes], name="input_y")
        self._zeros = tf.placeholder(tf.float32, shape=[None, self._classes])
        self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._global_step = tf.Variable(0, trainable=False)

    def _build_network(self, x, y, is_training):
        x_resh = tf.reshape(x, [-1, self._input_width, self._input_height, self._input_channels])
        # conv1_1 = conv_layer(x_resh, 3, 1, 128, is_training, name="conv1_1")
        # conv1_2 = conv_layer(conv1_1, 3, 1, 128, is_training, name="conv1_2")
        # pool1 = avg_pool_layer(conv1_2, 2, 2, name="pool1")
        #
        # conv2_1 = conv_layer(pool1, 3, 1, 256, is_training, name="conv2_1")
        # conv2_2 = conv_layer(conv2_1, 3, 1, 256, is_training, name="conv2_2")
        # pool2 = avg_pool_layer(conv2_2, 2, 2, name="pool2")
        #
        # conv3_1 = conv_layer(pool2, 3, 1, 512, is_training, name="conv3_1")
        # conv3_2 = conv_layer(conv3_1, 3, 1, 512, is_training, name="conv3_2")
        # conv3_3 = conv_layer(conv3_2, 3, 1, 512, is_training, name="conv3_3")
        # pool3 = avg_pool_layer(conv3_3, 2, 2, name="pool3")
        #
        # fc_in = tf.reshape(pool3, [-1, 3 * 3 * 512])
        # fc4 = fc_layer(fc_in, 4096, is_training, name="fc4", relu_flag=True)
        # dropout4 = tf.nn.dropout(fc4, self._keep_prob)
        #
        # fc5 = fc_layer(dropout4, 4096, is_training, name="fc5", relu_flag=True)
        # dropout5 = tf.nn.dropout(fc5, self._keep_prob)
        #
        # fc6 = fc_layer(dropout5, self._classes, is_training, name="fc6", relu_flag=False)

        conv1 = conv_layer(x_resh, 5, 1, 32, is_training, name="conv1")
        pool1 = avg_pool_layer(conv1, 2, 2, name="pool1")

        conv2 = conv_layer(pool1, 5, 1, 64, is_training, name="conv2")
        pool2 = avg_pool_layer(conv2, 2, 2, name="pool2")

        fc_in = tf.reshape(pool2, [-1, 5 * 5 * 64])
        fc3 = fc_layer(fc_in, 1024, is_training, name="fc3", relu_flag=True)
        dropout3 = tf.nn.dropout(fc3, self._keep_prob)

        fc4 = fc_layer(dropout3, self._classes, is_training, name="fc4", relu_flag=True)

        out = fc4
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=out))
        # accu = tf.reduce_mean(tf.cast(tf.equal(tf.round((tf.nn.sigmoid(out) - y) * 1.01), self._zeros), tf.float32))

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
        loss = weighted_loss(out, y, self._classes, self._loss_array)
        accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)), tf.float32))
        return loss, out, accu

    def _train_set(self, total_loss, global_step):
        loss_averages_op = add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(self._learning_rate)
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            loss, prediction, accu = self._build_network(self._x, self._y, self._is_training)
            train_op = self._train_set(loss, self._global_step)

            saver = tf.train.Saver()
            tf.add_to_collection('prediction', prediction)

            summary_op = tf.summary.merge_all()

            if self._start_step > 0:
                saver.restore(sess, CKPT_PATH)
            else:
                sess.run(tf.global_variables_initializer())

            # summary_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
            loss_pl = tf.placeholder(tf.float32)
            accu_pl = tf.placeholder(tf.float32)
            loss_summary = tf.summary.scalar("Average_loss", loss_pl)
            accu_summary = tf.summary.scalar("Prediction_accuracy", accu_pl)

            for step in range(self._start_step + 1, self._start_step + self._epoch_size + 1):
                print("Training epoch %d/%d" % (step, self._epoch_size))
                total_batch = len(self._raws)
                epoch_loss, epoch_accu = np.zeros(total_batch), np.zeros(total_batch)
                for bat in range(total_batch):
                    batch_xs = self._raws[bat]
                    batch_ys = self._labels[bat]
                    _, sum_str, pd, los, acu = sess.run([train_op, summary_op, prediction, loss, accu],
                                                   feed_dict={self._x: batch_xs, self._y: batch_ys,
                                                              self._keep_prob: self._keep_pb,
                                                              self._is_training: True,
                                                              self._zeros: np.zeros(batch_ys.shape)})
                    epoch_loss[bat], epoch_accu[bat], loss_str, accu_str = sess.run([loss_pl, accu_pl, loss_summary,
                                                                                     accu_summary],
                                                                                    feed_dict={loss_pl: los,
                                                                                               accu_pl: acu})
                    print("Training epoch %d/%d, batch %d/%d, loss %g, accuracy %g" %
                          (step, self._epoch_size, bat, total_batch, epoch_loss[bat], epoch_accu[bat]))

                print("Training epoch %d/%d finished, loss %g, accuracy %g" %
                      (step, self._epoch_size, np.mean(epoch_loss), np.mean(epoch_accu)))
                print("==============================================================")

                # summary_writer.add_summary(sum_str, step)
                # summary_writer.add_summary(loss_str, step)
                # summary_writer.add_summary(accu_str, step)

                if step % 1 == 0:
                    print("Testing epoch {0}".format(step))
                    test_batch = len(self._test_raws)
                    epoch_loss, epoch_accu = np.zeros(test_batch), np.zeros(test_batch)
                    for bat in range(test_batch):
                        batch_xs = self._test_raws[bat]
                        batch_ys = self._test_labels[bat]
                        pd, los, acu = sess.run([prediction, loss, accu],
                                                feed_dict={self._x: batch_xs, self._y: batch_ys,
                                                                       self._keep_prob: 1.0,
                                                                       self._is_training: False,
                                                                       self._zeros: np.zeros(batch_ys.shape)})
                        epoch_loss[bat], epoch_accu[bat], loss_str, accu_str = sess.run([loss_pl, accu_pl, loss_summary,
                                                                                         accu_summary],
                                                                                        feed_dict={loss_pl: los,
                                                                                                   accu_pl: acu})
                        print("Testing epoch %d/%d, batch %d/%d, loss %g, accuracy %g" %
                              (step, self._epoch_size, bat, test_batch, epoch_loss[bat], epoch_accu[bat]))

                    print("Testing epoch %d/%d finished, loss %g, accuracy %g" %
                          (step, self._epoch_size, np.mean(epoch_loss), np.mean(epoch_accu)))
                    print("==============================================================")

                    # cnt = int(self._test_raws.shape[0] / self._batch_size)
                    # losses, accu_total, iu, accu_class = np.zeros(cnt), np.zeros(cnt), np.zeros(cnt), np.zeros(
                    #     self._classes)
                    # for j in range(cnt):
                    #     x_test = [self._test_raws[i] for i in range(j * self._batch_size, (j + 1) * self._batch_size)]
                    #     y_test = [self._test_labels[i] for i in range(j * self._batch_size, (j + 1) * self._batch_size)]
                    #     pred, losses[j] = sess.run([prediction, loss],
                    #                                feed_dict={self._x: x_test, self._y: y_test, self._keep_prob: 1.0,
                    #                                           self._zeros: np.zeros(y_test.shape)})
                    #     accu_total[j], iu[j], accu = per_class_acc(pred, y_test)
                    #     accu_class += accu
                    # print("train %d, loss %g, accu %g, mean IU %g" %
                    #       (step, np.mean(losses), np.mean(accu_total), np.mean(iu)))
                    # for ii in range(self._classes):
                    #     print("\tclass # %d accuracy = %f " % (ii, accu_class[ii] / cnt))

                print("saving model.....")
                # saver.save(sess, CKPT_PATH)
                time.sleep(10)
                print("end saving....\n")
