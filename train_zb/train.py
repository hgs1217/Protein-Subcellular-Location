# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:52:18
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-20 15:30:29
# 
# Raw Dataset: 3 3000x3000 images -> 6 labels
# My Dataset: 236 samples, each contains 10000 32x32 patches -> 6 labels
# Training:
#   1. Select 1 positive and negative sample (20000 patches)
#   2. Pass to the network  (output: 20000 values)
#   3. Select the average of max 1000 to represent (2 values)
#   4. Loss = hingeloss


import tensorflow as tf

from config import config
from dataset import DataGenerator
from model import model_train, model_eval

def get_train_op():
    dataset = DataGenerator(
            sample_range=config.train_samples,      # we have 236 samples
            shuffle_samples=True,
            max_patches_per_sample=config.max_patches_per_sample
        ).make_dataset(
            split_pos_neg=True
        )

    pos, neg, label_index = dataset.make_one_shot_iterator().get_next()
    op, metrics = model_train(pos, neg, label_index, params={'n_candidates': config.n_candidates})
    return op, metrics

def get_eval_op():
    dataset = DataGenerator(
            sample_range=config.eval_samples,
            shuffle_samples=True,
            max_patches_per_sample=config.max_patches_per_sample
        ).make_dataset(
            split_pos_neg=False
        )

    X, Y = dataset.make_one_shot_iterator().get_next()
    metrics = model_eval(X, Y, params={'n_candidates': config.n_candidates})
    return metrics

with tf.Session() as sess:
    train_op, train_metrics = get_train_op()
    eval_metrics = get_eval_op()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "tb/model_e002.ckpt")

    n_epoches = 100
    n_train_steps = 10
    n_eval_steps = 5

    for epoch in range(n_epoches):
        epoch_loss = 0
        epoch_f1 = 0

        for step in range(n_train_steps):
            _, loss, pos_hist, neg_hist, ps, ns = sess.run([train_op, 
                train_metrics['loss'], train_metrics['pos_hist'], train_metrics['neg_hist'],
                train_metrics['pos_shape'], train_metrics['neg_shape']
            ])
            epoch_loss += loss

            pos_hist = "[" + ",".join([f"{x}" for x in pos_hist]) + "]"
            neg_hist = "[" + ",".join([f"{x}" for x in neg_hist]) + "]"

            # print(ps, ns)
            print(f"* epoch {epoch+1}/{n_epoches} step {step+1}/{n_steps}: loss={loss:.3f}, pos_hist={pos_hist}, neg_hist={neg_hist}")

        save_path = saver.save(sess, config.ckpt_dir/f"model_e{epoch:03d}.ckpt")
        print("Model saved in path: %s" % save_path)

        for i in range(n_eval_steps):
            metrics = sess.run(eval_metrics)
            metrics_ = {
                'pred': "[" + ",".join([f"{x:.2f}" for x in metrics['pred']]) + "]",
                'labels': metrics['labels'],
                'prec': f"{metrics['precision']:.4f}",
                'reca': f"{metrics['recall']:.4f}",
                'f1': f"{metrics['f1score']:.4f}"
            }
            epoch_f1 += f1score

            log_str = " ".join([f"{k}={v}" for k,v in metrics_.items()])
            print(f"* epoch {epoch+1}/{n_epoches} sample {i}: " + log_str)

        epoch_loss /= n_train_steps
        epoch_f1 /= n_eval_steps

        print(f"** epoch {epoch+1}/{n_epoches}: loss={epoch_loss:.4f} f1={epoch_f1:.4f}")
        merged = sess.run(tf.summary.merge_all())
        writer.add_summary(summary, i)


