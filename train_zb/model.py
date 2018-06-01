# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:56:30
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-01 16:13:54

import tensorflow as tf
import numpy as np

def model(features, labels, mode, params):
    """
    Interface between logical codes and tensorflow estimator
    The signature of this function is required by tensorflow

    @params: features: a ndarray in the shape of (1, n_patches, width, height)
    @params: labels:  a ndarray in the shape of (1, n_patches, )
    """
    print(labels)
    if mode is tf.estimator.ModeKeys.PREDICT:

        net_out = net(features, training=False)
        return model_predict(net_out, params)

    else:

        net_out = net(features, training=True)

        if mode == tf.estimator.ModeKeys.EVAL:  return model_eval(net_out, labels, params)
        if mode == tf.estimator.ModeKeys.TRAIN: return model_train(net_out, labels, params)


def net(val, training=True):
    val = tf.reshape(val, [2, -1, 32 * 32])
    val = tf.layers.dense(val, units=1, activation=tf.nn.sigmoid)
    val = tf.reshape(val, [2, -1])
    return val

def model_predict(net_out, params):
    raise NotImplementedError
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=prediction)

def model_eval(net_out, labels, params):
    raise NotImplementedError
    # pred = tf.argmax(net_out, axis=-1, name='pred')
    # accuracy = tf.metrics.accuracy(labels=labels, predictions=pred, name='accuracy')
    # tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops={'accuracy': accuracy})

def model_train(net_out, labels, params):
    pos_data_out, neg_data_out = net_out[0], net_out[1]
    pos_max = tf.nn.top_k(pos_data_out, k=1).values
    neg_max = tf.nn.top_k(neg_data_out, k=1).values
    loss = 1 - (pos_max - neg_max)
    tf.identity(loss, name='loss')
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)
