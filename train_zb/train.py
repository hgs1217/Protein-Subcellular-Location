# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:52:18
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-21 13:40:29
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
    ops = model_train(lhs, lhs_label, rhs, rhs_label, params={'n_candidates': config.n_candidates})
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
    ops = model_eval(X, Y, params={'n_candidates': config.n_candidates})
    return ops

def float_list_to_str(lst, format_str='{:.2f}'):
    ret = ','.join([format_str.format(x) for x in lst])
    return '[' + ret  + ']'

class Stat:
    def __init__(self, update_type, init_val, **kwargs):
        self.val = init_val
        self.type = update_type

        if update_type == 'exp':
            self.decay = kwargs.get('decay', 0.9)
        elif update_type == 'avg':
            self.count = 0

        self.format = kwargs.get('format', '{:.2f}') # for output

    def update(self, val):
        if self.type == 'exp':
            self.val -= (self.val - val) * (1 - self.decay)
        elif self.type == 'avg':
            self.val -= (self.val - val) / (1 + self.count)
            self.count += 1

    def __repr__(self):
        if isinstance(self.val, (list, tuple, np.ndarray)):
            return float_list_to_str(self.val, format_str=self.format)
        else:
            return self.format.format(self.val)


with tf.Session() as sess:
    train_ops = get_train_op()
    eval_ops = get_eval_op()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "tb/model_e002.ckpt")

    n_epoches = 100
    n_train_steps = 1
    n_eval_steps = 3

    train_loss = Stat('exp', 0, decay=0.9, format='{:.4f}')
    train_acc  = Stat('exp', np.zeros(config.n_labels), decay=0.9, format='{:.3f}')
    eval_f1   = Stat('avg', 0, format='{:.4f}')
    eval_acc  = Stat('avg', np.zeros(config.n_labels), format='{:.3f}')

    for epoch in range(n_epoches):

        tic = time.time()
        for step in range(n_train_steps):
            result = sess.run(train_ops)

            train_loss.update(result['loss'])
            train_acc.update((result['lhs_pred'] == result['lhs_label']).astype(np.float))
            train_acc.update((result['rhs_pred'] == result['rhs_label']).astype(np.float))

            log_str = f"* epoch {epoch+1}/{n_epoches} step {step+1}/{n_train_steps}:"
            for k,v in {
                'loss': train_loss,
                'lhs_pred': float_list_to_str(result['lhs_pred']),
                'rhs_pred': float_list_to_str(result['rhs_pred'])
            }.items():
                log_str += f" {k}={v}"
            print(log_str)
        train_speed = (time.time() - tic) / n_train_steps

        # save_path = saver.save(sess, config.ckpt_dir/f"model_e{epoch:03d}.ckpt")
        # print("Model saved in path: %s" % save_path)

        tic = time.time()
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
        eval_speed = (time.time() - tic) / n_eval_steps

        print(f"\n** epoch {epoch+1}/{n_epoches}: train_loss={train_loss} train_acc={train_acc} eval_f1={eval_f1} eval_acc={eval_acc} speed={train_speed:.2f}sec/train {eval_speed:.2f}sec/eval\n")


