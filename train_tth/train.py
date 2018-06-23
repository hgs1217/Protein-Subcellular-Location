# -*- coding: utf-8 -*-

from train_tth.cnn import CNN
import os
from train_tth.data_construct import data_construction
from train_tth.config import TEST_RATIO


def train(classfier_num=0, simple=False, start_step=0, epoch_size=100, keep_pb=0.5, learning_rate=0.001,
          detail_log=False,
          open_summary=False, new_ckpt_internal=0, gpu=True, ):
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use cpu only

    raws, labels, test_raws, test_labels = data_construction(classfier_num, TEST_RATIO)

    # print(raws.shape)
    # print(labels.shape)
    # print(loss_array)

    if simple:
        raws = raws[:50]
        labels = labels[:50]
        test_raws = test_raws[:10]
        test_labels = test_labels[:10]

    cnn = CNN(classfier_num, raws, labels, test_raws, test_labels, epoch_size=epoch_size,
              start_step=start_step, keep_pb=keep_pb, learning_rate=learning_rate,
              detail_log=detail_log, open_summary=open_summary, new_ckpt_internal=new_ckpt_internal,
              )
    cnn.train()
