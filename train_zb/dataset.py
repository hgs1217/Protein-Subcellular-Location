# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:56:34
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-05-29 12:35:00

import _import_helper
from data_process.image_preprocessor import ImagePreprocessor

import numpy as np

def get_data():
    """
    Wrapper around ImagePreprocessor
    Goodbye OOP
    """
    import os
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data_process', 'HPA_ieee')
    p = ImagePreprocessor(base_dir=base_dir)

    label_dict = {}
    label_dict_max_size =  30
    for label1, label2, imgs in p.get_dataset_full(data_selection='all'):
        labels = set(filter(None, (label1 + ';' + label2).split(';')))
        label_vec = [0] * label_dict_max_size
        for l in labels:
            label_dict.setdefault(l, len(label_dict))
            label_vec[label_dict[l]] = 1
        for img in imgs:
            print('yielded')
            yield img.astype(np.float32), label_vec


import tensorflow as tf

def make_dataset():
    def feed_dataset(batch_size, training):
        ds = tf.data.Dataset.from_generator(
                get_data,
                (tf.float32, tf.int64),
                (tf.TensorShape([3000, 3000, 3]), tf.TensorShape([None]))
            )
        if training:
            ds = ds.shuffle(100)
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        return ds

    return feed_dataset
