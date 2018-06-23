# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def add_loss_summaries(total_loss):
    """
        Loss summaries used in tensorboard
        :param name: float
        :return: loss_averages_op
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def cal_loss(lgts, lbs):
    """
        Weight loss function calculation. We have loss weights to different labels
        :param lgts: tensor
                    Prediction outputs given by the model
        :param lbs: tensor
                    Labels given as prior
        :param num_classes: int
                    Label class number
        :param loss_array: list
                    Loss weights array
        :return:
    """
    with tf.name_scope('loss'):
        epsilon = tf.constant(value=1e-10)
        labels = lbs
        logits = lgts + epsilon

        cross_entropy = -tf.reduce_sum(labels * tf.log(tf.nn.softmax(logits) + epsilon), axis=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy', axis=0)
        return cross_entropy_mean


def conv_layer(x, ksize, stride, feature_num, is_training, name=None, padding="SAME", groups=1):
    """
        Convolutional layer.
        :param x: tensor
                Input tensor
        :param ksize: int
                Convolution kernel size
        :param stride: int
                Convolution stride length
        :param feature_num: int
                Output feature number
        :param is_training: boolean
                Whether it is in the training step
        :param name: string
                Layer name
        :param padding: string
                Convolutional padding mode
        :param groups: int
                Convolutional groups needed to split
        :return:
    """
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
    """
        Average pooling layer
        :param x: tensor
                Input tensor
        :param ksize: int
                Convolution kernel size
        :param stride: int
                Convolution stride length
        :param name: string
                Layer name
        :param padding: string
                Convolutional padding mode
        :return:
    """
    return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)


def fc_layer(x, feature_num, is_training, name=None, relu_flag=True):
    """
        Full connected layer
        :param x: tensor
                Input tensor
        :param feature_num: int
                Output feature number
        :param is_training: boolean
                Whether it is in the training step
        :param name: string
                Layer name
        :param relu_flag: boolean
                Whether this layer contains relu activation
        :return:
    """
    with tf.variable_scope(name) as scope:
        w = variable_with_weight_decay('w', shape=[x.get_shape()[-1], feature_num],
                                       initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b", [feature_num], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(tf.matmul(x, w), b)
        norm = batch_norm_layer(bias, is_training)
        return tf.nn.relu(norm) if relu_flag else norm
