# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:52:18
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-22 09:25:50
# 
# Raw Dataset: 3 3000x3000 images -> 6 labels
# My Dataset: 236 samples, each contains 10000 32x32 patches -> 6 labels
# Training:
#   1. Select 1 lhs and rhs sample (20000 patches)
#   2. Pass to the network  (output: 20000 values)
#   3. Select the average of max 1000 to represent (2 values)
#   4. Loss = hingeloss

import time
import tensorflow as tf
import numpy as np

import config
from util import Stat, float_list_to_str, Timer
from dataset import DataGenerator
from model import model_train, model_eval

def get_train_op_old():
    dataset = DataGenerator(
            sample_range=config.train_samples,      # we have 236 samples
            shuffle_samples=True,
            max_patches_per_sample=config.max_patches_per_sample
        ).make_dataset(
            split_lhs_rhs=True
        )

    lhs, lhs_label, rhs, rhs_label = dataset.make_one_shot_iterator().get_next()
    ops = model_train(lhs, lhs_label, rhs, rhs_label, params={'n_candidates': tf.constant(config.n_candidates)})
    return ops

def get_train_op():
    dg = DataGenerator(
            sample_range=config.train_samples,
            shuffle_samples=True,
            max_patches_per_sample=config.max_patches_per_sample
        )
    ds = dg.make_dataset(split_lhs_rhs=False)
    label_weights = dg.get_label_weights_from_dumped()
    print("Label weights used: ", label_weights)
    label_weights = tf.constant(label_weights, dtype=tf.float32)

    X, Y = ds.make_one_shot_iterator().get_next()
    ops = model_train(X, Y, params={
        'n_candidates': tf.constant(config.n_candidates),
        'label_weights': label_weights
    })
    return ops

def get_eval_op():
    dataset = DataGenerator(
            sample_range=config.eval_samples,
            shuffle_samples=True,
            max_patches_per_sample=config.max_patches_per_sample
        ).make_dataset(
            split_lhs_rhs=False
        )

    X, Y = dataset.make_one_shot_iterator().get_next()
    ops = model_eval(X, Y, params={'n_candidates': tf.constant(config.n_candidates)})
    return ops

n_epoches = 200
n_train_steps = 100
n_eval_steps = 25

train_loss = Stat('exp', 0, decay=0.9, format='{:.4f}')
train_acc  = Stat('exp', np.zeros(config.n_labels), decay=0.9, format='{:.3f}')
eval_f1   = Stat('avg', 0, format='{:.4f}')
eval_acc  = Stat('avg', np.zeros(config.n_labels), format='{:.3f}')
train_timer = Timer()
eval_timer = Timer()

# hooks
def post_train_old(result, epoch, step):
    train_loss.update(result['loss'])
    train_acc.update((result['lhs_pred'] == result['lhs_label']).astype(np.float))
    train_acc.update((result['rhs_pred'] == result['rhs_label']).astype(np.float))

    log_str = f"* epoch {epoch+1}/{n_epoches} step {step+1}/{n_train_steps}:"
    for k,v in {
        'loss': f"{result['loss']:.4f}",
        'lhs_prob': float_list_to_str(result['lhs_prob']),
        'rhs_prob': float_list_to_str(result['rhs_prob'])
    }.items():
        log_str += f" {k}={v}"
    print(log_str)

def post_train(result, epoch, step):
    train_loss.update(result['loss'])
    train_acc.update((result['pred'] == result['label']).astype(np.float))

    log_str = f"* epoch {epoch+1}/{n_epoches} step {step+1}/{n_train_steps}:"
    for k,v in {
        'loss': f"{result['loss']:.4f}",
        'prob': float_list_to_str(result['prob']),
    }.items():
        log_str += f" {k}={v}"
    print(log_str)

def post_eval(result, epoch, step):
    eval_f1.update(result['f1score'])
    eval_acc.update((result['pred'] == result['labels']).astype(np.float))

    log_str = f"* epoch {epoch+1}/{n_epoches} sample {i}:"
    for k,v in {
        'prob': float_list_to_str(result['prob']),
        'pred': result['pred'],
        'labels': result['labels'],
        'prec': f"{result['precision']:.4f}",
        'reca': f"{result['recall']:.4f}",
        'f1': f"{result['f1score']:.4f}"
    }.items():
        log_str += f" {k}={v}"
    print(log_str)

def post_epoch(epoch):
    eval_acc.reset()
    eval_f1.reset()

    epoch_log_str = f"** epoch {epoch+1}/{n_epoches}:"
    for k,v in {
        "train_loss": train_loss,
        "train_acc": train_acc, 
        "eval_f1": eval_f1,
        "eval_acc": eval_acc,
        "speed": f"{train_timer.time_sec:.2f}sec/train {eval_timer.time_sec:.2f}sec/eval"
    }.items():
        epoch_log_str += f" {k}={v}"
    print("\n" + epoch_log_str + "\n")

# Mainloop
with tf.Session() as sess:
    train_ops = get_train_op()
    eval_ops = get_eval_op()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "tb/model_e002.ckpt")

    for epoch in range(n_epoches):
        with train_timer(rounds=n_train_steps):
            for step in range(n_train_steps):
                result = sess.run(train_ops)
                post_train(result, epoch=epoch, step=step)

        save_path = saver.save(sess, config.ckpt_dir/f"model_e{epoch:03d}.ckpt")
        print("Model saved in path: %s" % save_path)

        with eval_timer(rounds=n_eval_steps):
            for i in range(n_eval_steps):
                result = sess.run(eval_ops)
                post_eval(result, epoch=epoch, step=step)

        post_epoch(epoch)