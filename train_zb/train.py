# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:52:18
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-21 14:07:04
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

from config import config
from util import Stat, float_list_to_str, Timer
from dataset import DataGenerator
from model import model_train, model_eval

def get_train_op():
    dataset = DataGenerator(
            sample_range=config.train_samples,      # we have 236 samples
            shuffle_samples=False,
            max_patches_per_sample=config.max_patches_per_sample
        ).make_dataset(
            split_lhs_rhs=True
        )

    lhs, lhs_label, rhs, rhs_label = dataset.make_one_shot_iterator().get_next()
    ops = model_train(lhs, lhs_label, rhs, rhs_label, params={'n_candidates': tf.constant(config.n_candidates)})
    return ops

def get_eval_op():
    dataset = DataGenerator(
            sample_range=config.eval_samples,
            shuffle_samples=False,
            max_patches_per_sample=config.max_patches_per_sample
        ).make_dataset(
            split_lhs_rhs=False
        )

    X, Y = dataset.make_one_shot_iterator().get_next()
    ops = model_eval(X, Y, params={'n_candidates': tf.constant(config.n_candidates)})
    return ops

with tf.Session() as sess:
    train_ops = get_train_op()
    eval_ops = get_eval_op()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "tb/model_e002.ckpt")

    n_epoches = 100
    n_train_steps = 10
    n_eval_steps = 30

    train_loss = Stat('exp', 0, decay=0.9, format='{:.4f}')
    train_acc  = Stat('exp', np.zeros(config.n_labels), decay=0.9, format='{:.3f}')
    eval_f1   = Stat('avg', 0, format='{:.4f}')
    eval_acc  = Stat('avg', np.zeros(config.n_labels), format='{:.3f}')
    train_timer = Timer()
    eval_timer = Timer()

    for epoch in range(n_epoches):

        with train_timer(rounds=n_train_steps):
            for step in range(n_train_steps):
                result = sess.run(train_ops)

                train_loss.update(result['loss'])
                train_acc.update((result['lhs_pred'] == result['lhs_label']).astype(np.float))
                train_acc.update((result['rhs_pred'] == result['rhs_label']).astype(np.float))

                log_str = f"* epoch {epoch+1}/{n_epoches} step {step+1}/{n_train_steps}:"
                for k,v in {
                    'loss': f"{result['loss']:.4f}",
                    'lhs_pred': float_list_to_str(result['lhs_pred']),
                    'rhs_pred': float_list_to_str(result['rhs_pred'])
                }.items():
                    log_str += f" {k}={v}"
                print(log_str)

        save_path = saver.save(sess, config.ckpt_dir/f"model_e{epoch:03d}.ckpt")
        print("Model saved in path: %s" % save_path)

        with eval_timer(rounds=n_eval_steps):
            for i in range(n_eval_steps):
                result = sess.run(eval_ops)
                eval_f1.update(result['f1score'])
                eval_acc.update((result['pred'] == result['labels']).astype(np.float))

                log_str = f"* epoch {epoch+1}/{n_epoches} sample {i}:"
                for k,v in {
                    'logits': float_list_to_str(result['logits']),
                    'pred': result['pred'],
                    'labels': result['labels'],
                    'prec': f"{result['precision']:.4f}",
                    'reca': f"{result['recall']:.4f}",
                    'f1': f"{result['f1score']:.4f}"
                }.items():
                    log_str += f" {k}={v}"
                print(log_str)


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
