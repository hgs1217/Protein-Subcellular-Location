# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:56:30
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-05-29 12:35:12

import tensorflow as tf
import numpy as np

def model(features, labels, mode, params):
    """
    Interface between logical codes and tensorflow estimator
    The signature of this function is required by tensorflow
    """
    if mode is tf.estimator.ModeKeys.PREDICT:

        net_out = net(features, training=False)
        return model_predict(net_out, params)

    else:

        net_out = net(features, training=True)

        if mode == tf.estimator.ModeKeys.EVAL:  return model_eval(net_out, labels, params)
        if mode == tf.estimator.ModeKeys.TRAIN: return model_train(net_out, labels, params)


def net(inputs, training=True):
    ret = inputs
    ret = tf.layers.flatten(ret)
    ret = tf.layers.dense(ret, units=30, activation=tf.nn.sigmoid)
    return ret

def model_predict(net_out, params):
    if params['model'] == 'super-bal':
        pred = tf.nn.softmax(net_out)[:, 1]
        prediction = { 'pred': pred }
    elif params['model'] == 'super-ext':
        pred = tf.nn.softmax(net_out)[:, :, :, :, 1]
        prediction = { 'pred': pred }

    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=prediction)

def model_eval(net_out, labels, params):
    pred = tf.argmax(net_out, axis=-1, name='pred')
    accuracy = tf.metrics.accuracy(labels=labels, predictions=pred, name='accuracy')
    tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops={'accuracy': accuracy})

def model_train(net_out, labels, params):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=net_out)
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)
