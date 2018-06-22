# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:56:30
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-22 10:47:22

import tensorflow as tf
import numpy as np
from itertools import product

import config

def net(val, training=True):
    conv = tf.layers.conv2d
    pool = tf.layers.average_pooling2d
    dropout = tf.layers.dropout
    dense = tf.layers.dense

    with tf.variable_scope('Net', reuse=tf.AUTO_REUSE):
        val =  tf.expand_dims(val, axis=-1)  # insert channel dim -> N  x 32 x 32 x 1
        val = conv(val, filters=32, kernel_size=5, strides=1, padding='same')   # -> N x 32 x 32 x 32
        val = pool(val, pool_size=2, strides=2)                                 # -> N x 16 x 16 x 32
        val = conv(val, filters=64, kernel_size=5, strides=1, padding='same')   # -> N x 16 x 16 x 64
        val = pool(val, pool_size=2, strides=2)                                 # -> N x 8 x 8 x 64
        val = conv(val, filters=128, kernel_size=5, strides=1)                  # -> N x 4 x 4 x 128
        val = tf.layers.flatten(val)                                            # -> N x (4 * 4 * 128)

        def dnn(val):
            val = dropout(val, rate=0.5, training=training)
            val = dense(val, units=1024, activation=tf.nn.relu)
            val = dense(val, units=256,  activation=tf.nn.sigmoid)             # -> N x 1
            val = dense(val, units=1,   activation=tf.nn.sigmoid)             # -> N x 1
            val = tf.reshape(val, [-1])
            return val

        heads = tf.transpose(tf.stack([dnn(val) for _ in range(6)]))    # -> N x 6
    return heads

def try_most_top_k(x, k):
    k_ = tf.minimum(tf.shape(x)[0], k)
    return tf.nn.top_k(x, k=k_)

def hard_threshold(prob):
    pred = tf.where(prob < 0.5, tf.zeros(tf.shape(prob)), tf.ones(tf.shape(prob)))
    return tf.cast(pred, tf.int64)

def model_train_old(lhs_features, lhs_label, rhs_features, rhs_label, params):
    assert len(lhs_features.shape) == len(rhs_features.shape) == 3

    lhs_out = net(lhs_features)
    lhs_top = try_most_top_k(lhs_out, k=params['n_candidates']).values
    lhs_prob = tf.reduce_mean(lhs_top, axis=-1)
    lhs_pred = hard_threshold(lhs_prob)

    rhs_out = net(rhs_features)
    rhs_top = try_most_top_k(rhs_out, k=params['n_candidates']).values
    rhs_prob = tf.reduce_mean(rhs_top, axis=-1)
    rhs_pred = hard_threshold(rhs_prob)

    # lhs_top = tf.reduce_mean(tf.nn.top_k(lhs_out, k=params['n_candidates']).values)
    # rhs_top = tf.reduce_mean(tf.nn.top_k(rhs_out, k=params['n_candidates']).values)
    loss_mask = tf.cast(tf.not_equal(lhs_label, rhs_label), tf.float32)

    loss_lhs = lhs_prob - rhs_prob
    loss_rhs = rhs_prob - lhs_prob
    loss = tf.where(tf.equal(lhs_label, 1), 1 - (lhs_prob - rhs_prob), 1 - (rhs_prob - lhs_prob))

    loss *= loss_mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(loss_mask)

    optimizer = tf.train.AdagradOptimizer(learning_rate=config.learning_rate)
    opt_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    ops = {
        'lhs_prob': lhs_prob,
        'lhs_pred': lhs_pred,
        'lhs_label': lhs_label,
        'rhs_prob': rhs_prob,
        'rhs_pred': rhs_pred,
        'rhs_label': rhs_label,
        'loss': loss,
        'opt_op': opt_op
    }

    return ops

def model_train(features, labels, params):
    assert len(features.shape) == 3

    out = net(features) # N x 6
    pred = hard_threshold(out)

    prob_softmax = tf.stack([1-out, out], axis=-1)  # shape = N x 6 x 2

    loss = tf.losses.sparse_softmax_cross_entropy(labels, prob_softmax, reduction=tf.losses.Reduction.NONE)  # shape = N x 6
    pos_weights = params['label_weights'][1]    # shape = 6
    neg_weights = params['label_weights'][0]    # shape = 6
    labels = tf.cast(labels, tf.float32)
    weights = labels * pos_weights + (1 - labels) * neg_weights
    loss_weighted = tf.reduce_mean(loss * weights)

    optimizer = tf.train.AdagradOptimizer(learning_rate=config.learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)
    opt_op = optimizer.minimize(loss_weighted, global_step=tf.train.get_global_step())
    ops = {
        'prob': out,
        'pred': pred,
        'labels': labels,
        'loss': loss_weighted,
        'opt_op': opt_op
    }

    return ops

def model_eval(features, labels, params):
    assert len(features.shape) == 3

    net_out = net(features, training=False) # N x 6
    pred = hard_threshold(net_out)

    TP = tf.equal(pred, 1) & tf.equal(labels, 1)
    TN = tf.equal(pred, 0) & tf.equal(labels, 0)
    FP = tf.equal(pred, 1) & tf.equal(labels, 0)
    FN = tf.equal(pred, 0) & tf.equal(labels, 1)

    TP, TN, FP, FN = (tf.reduce_sum(tf.cast(x, tf.float32), axis=-1) for x in [TP, TN, FP, FN])

    zeros = tf.zeros(tf.shape(TP), dtype=tf.float32)
    ones  = tf.ones (tf.shape(TP), dtype=tf.float32)
    precision = tf.where(TP + FP < 0.1, zeros, TP / (TP + FP)) + tf.constant(1e-5)
    recall    = tf.where(TP + FN < 0.1, ones, TP / (TP + FN)) + tf.constant(1e-5)
    f1score   = 2 / (1 / precision + 1 / recall)

    ops = {
        'prob': net_out,
        'pred': pred,
        'labels': labels,
        'precision': tf.reduce_mean(precision),
        'recall': tf.reduce_mean(recall),
        'f1score': tf.reduce_mean(f1score)
    }
    return ops
