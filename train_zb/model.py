# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:56:30
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-20 13:26:56

import tensorflow as tf
import numpy as np
from itertools import product


def net(val, training=True):
    conv = tf.layers.conv2d
    pool = tf.layers.average_pooling2d
    dense = tf.layers.dense

    val =  tf.expand_dims(val, axis=-1)  # insert channel dim -> N  x 32 x 32 x 1
    val = conv(val, filters=32, kernel_size=5, strides=1, padding='same')   # -> N x 32 x 32 x 32
    val = pool(val, pool_size=2, strides=2)                                 # -> N x 16 x 16 x 32
    val = conv(val, filters=64, kernel_size=5, strides=1, padding='same')   # -> N x 16 x 16 x 64
    val = pool(val, pool_size=2, strides=2)                                 # -> N x 8 x 8 x 64
    val = conv(val, filters=128, kernel_size=8, strides=1)                  # -> N x 1 x 1 x 128
    val = tf.layers.flatten(val)                                           # -> N x 128

    def dnn(val):
        val = dense(val, units=64, activation=tf.nn.relu)
        val = dense(val, units=16, activation=tf.nn.relu)
        val = dense(val, units=1,  activation=tf.nn.sigmoid)             # -> N x 1
        val = tf.reshape(val, [-1])
        return val

    heads = tf.stack([dnn(val) for _ in range(6)])    # -> 6 x N
    return heads

def model_train(pos_features, neg_features, label_index, params):
    assert len(pos_features.shape) == len(neg_features.shape) == 3

    pos_out = net(pos_features)[label_index]
    neg_out = net(neg_features)[label_index]

    pos_top = tf.reduce_mean(tf.nn.top_k(pos_out, k=params['n_candidates']).values)
    neg_top = tf.reduce_mean(tf.nn.top_k(neg_out, k=params['n_candidates']).values)
    loss = 1 - (pos_top - neg_top)

    metrics = {
        'pos_mean': tf.reduce_mean(pos_out),
        'pos_max': tf.reduce_max(pos_out),
        'pos_min': tf.reduce_min(pos_out),
        'pos_hist': tf.histogram_fixed_width(pos_out, value_range=[0, 1], nbins=10),
        'pos_top': pos_top,
        'neg_mean': tf.reduce_mean(neg_out),
        'neg_max': tf.reduce_max(neg_out),
        'neg_min': tf.reduce_min(neg_out),
        'neg_hist': tf.histogram_fixed_width(neg_out, value_range=[0, 1], nbins=10),
        'neg_top': neg_top,
        'loss': loss,
        'pos_shape': tf.shape(pos_features),
        'neg_shape': tf.shape(neg_features)
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

    TP, TN, FP, FN = (tf.reduce_sum(tf.cast(x, tf.int64)) for x in [TP, TN, FP, FN])

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1score   = 2 * precision * recall / (precision + recall)

    metrics = {
        'pred': net_out_top,
        'labels': labels,
        'precision': precision,
        'recall': recall,
        'f1score': f1score
    }
    metrics = {k: tf.identity(v, name=k) for k,v in metrics.items()}

    return metrics
