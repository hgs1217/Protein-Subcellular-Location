# -*- coding: utf-8 -*-  
"""
@author: Suibin Sun
@file: train.py
@time: 2018/6/20 18:43
"""

import _import_helper
from train.cnn import CNN
from data_process.data_preprocessor import generate_patch, data_construction, data_construction_v2

import os

TARGET_LABELS = [0, 1, 2, 3, 4, 5]
TEST_RATIO = 8 / 9


def train(simple=False, start_step=0, epoch_size=100, keep_pb=0.5, learning_rate=0.001, detail_log=False,
          open_summary=False, new_ckpt_internal=0, batch_size=None, gpu=True, network_mode=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if gpu else '-1'

    raws, labels, test_raws, test_labels, loss_array = \
        data_construction(TARGET_LABELS, TEST_RATIO) if batch_size is None \
            else data_construction_v2(TARGET_LABELS, batch_size, TEST_RATIO)

    print(raws[0].shape)
    print(labels[0].shape)
    print(loss_array)

    if simple:
        raws = raws[:50]
        labels = labels[:50]
        test_raws = test_raws[:10]
        test_labels = test_labels[:10]

    cnn = CNN(raws, labels, test_raws, test_labels, epoch_size=epoch_size, loss_array=loss_array,
              start_step=start_step, keep_pb=keep_pb, learning_rate=learning_rate,
              detail_log=detail_log, open_summary=open_summary, new_ckpt_internal=new_ckpt_internal,
              network_mode=network_mode)
    cnn.train()


if __name__ == '__main__':
    train(
        simple=True,
        start_step=0,
        epoch_size=100,
        keep_pb=0.5,
        learning_rate=0.001,
        detail_log=False,
        open_summary=False,
        new_ckpt_internal=0,
        batch_size=None,
        gpu=True,
        network_mode='lr'
    )
