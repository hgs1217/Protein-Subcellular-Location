# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:56:30
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-20 23:10:44

import tensorflow as tf
import numpy as np
from itertools import product


def net(val, training=True):
    conv = tf.layers.conv2d
    pool = tf.layers.average_pooling2d
    dropout = tf.layers.dropout
    dense = tf.layers.dense

    val =  tf.expand_dims(val, axis=-1)  # insert channel dim -> N  x 32 x 32 x 1
    val = conv(val, filters=32, kernel_size=5, strides=1, padding='same')   # -> N x 32 x 32 x 32
    val = pool(val, pool_size=2, strides=2)                                 # -> N x 16 x 16 x 32
    val = conv(val, filters=64, kernel_size=5, strides=1, padding='same')   # -> N x 16 x 16 x 64
    val = pool(val, pool_size=2, strides=2)                                 # -> N x 8 x 8 x 64
    val = conv(val, filters=128, kernel_size=8, strides=1)                  # -> N x 1 x 1 x 128
    val = tf.layers.flatten(val)                                           # -> N x 128

    def dnn(val):
        val = dropout(val, rate=0.5, training=training)
        val = dense(val, units=64, activation=tf.nn.relu)
        val = dense(val, units=16, activation=tf.nn.relu)
        val = dense(val, units=1,  activation=tf.nn.sigmoid)             # -> N x 1
        val = tf.reshape(val, [-1])
        return val

    heads = tf.stack([dnn(val) for _ in range(6)])    # -> 6 x N
    return heads

def model_train(lhs_features, lhs_label, rhs_features, rhs_label, params):
    assert len(lhs_features.shape) == len(rhs_features.shape) == 3

    lhs_out = net(lhs_features)
    lhs_top = tf.nn.top_k(lhs_out, k=params['n_candidates']).values
    lhs_pred = tf.reduce_mean(lhs_top, axis=-1)

    rhs_out = net(rhs_features)
    rhs_top = tf.nn.top_k(rhs_out, k=params['n_candidates']).values
    rhs_pred = tf.reduce_mean(rhs_top, axis=-1)

    # lhs_top = tf.reduce_mean(tf.nn.top_k(lhs_out, k=params['n_candidates']).values)
    # rhs_top = tf.reduce_mean(tf.nn.top_k(rhs_out, k=params['n_candidates']).values)
    loss_mask = tf.cast(tf.not_equal(lhs_label, rhs_label), tf.float32)

    loss_lhs = lhs_pred - rhs_pred
    loss_rhs = rhs_pred - lhs_pred
    loss = tf.where(tf.equal(lhs_label, 1), 1 - (lhs_pred - rhs_pred), 1 - (rhs_pred - lhs_pred))

    loss *= loss_mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(loss_mask)

    metrics = {
        'lhs_pred': lhs_pred,
        'rhs_pred': rhs_pred,
        'loss': loss,
    }
    metrics = {k: tf.identity(v, name=k) for k,v in metrics.items()}

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return train_op, metrics

def model_eval(features, labels, params):
    assert len(features.shape) == 3

    net_out = net(features, training=False)
    net_out_top = tf.nn.top_k(net_out, k=params['n_candidates']).values
    net_out_top = tf.reduce_mean(net_out_top, axis=-1)

    pred = tf.where(net_out_top < 0.5, tf.zeros(tf.shape(net_out_top)), tf.ones(tf.shape(net_out_top)))
    pred = tf.cast(pred, tf.int64)

    TP = tf.equal(pred, 1) & tf.equal(labels, 1)
    TN = tf.equal(pred, 0) & tf.equal(labels, 0)
    FP = tf.equal(pred, 1) & tf.equal(labels, 0)
    FN = tf.equal(pred, 0) & tf.equal(labels, 1)

    TP, TN, FP, FN = (tf.reduce_sum(tf.cast(x, tf.float32)) for x in [TP, TN, FP, FN])

    precision = tf.where(TP + FP < 0.1, tf.constant(1.0), TP / (TP + FP)) + tf.constant(1e-5)
    recall    = tf.where(TP + FN < 0.1, tf.constant(1.0), TP / (TP + FN)) + tf.constant(1e-5)
    f1score   = 2 / (1 / precision + 1 / recall)

    metrics = {
        'pred': net_out_top,
        'labels': labels,
        'precision': precision,
        'recall': recall,
        'f1score': f1score
    }
    metrics = {k: tf.identity(v, name=k) for k,v in metrics.items()}

    return metrics
