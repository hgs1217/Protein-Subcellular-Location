# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:56:34
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-17 22:58:32

import numpy as np
import itertools

import _import_helper
from data_process.image_preprocessor import ImagePreprocessor
from config import config
from cut_img import cut_img

def branch_tee(generator, condition_func):
    gen_true, gen_false = itertools.tee(generator)
    gen_true  = filter(lambda val: condition_func(val), gen_true)
    gen_false = filter(lambda val: not condition_func(val), gen_false)
    return gen_true, gen_false


class DataGenerator:
    def __init__(self):
        self._reload()

    def img_to_feature(self, img):
        # img: width x height x 3
        patches, positions, demo = cut_img(img)
        patches, positions = map(np.stack, (patches, positions))
        patches = np.mean(patches.astype(np.float32), axis=-1)   # convert to gray scale
        # patches: N x width x height
        # positions: N x 2
        return patches

    def dump(self):
        import pickle
        import cv2

        _data = ImagePreprocessor(base_dir=config.base_dataset_dir).get_dataset_full(data_selection='all', label_type='non-str')

        def get_sample(labels, imgs):
            patches, positions, demos = zip(*[cut_img(img) for img in imgs])
            patches = np.vstack([np.stack(p) for p in patches])
            patches = np.mean(patches.astype(np.float32), axis=-1)   # convert to gray scale
            return patches, labels, demos

        for ind, (label1, label2, imgs) in enumerate(_data):
            labels = set(label1) | set(label2)
            patches, labels, demos = get_sample(labels, imgs)

            with open(config.dataset_dir + f"/sample{ind:04d}", 'bw') as f:
                pickle.dump({'patches': patches, 'labels': labels}, f)

            for i, demo in enumerate(demos):
                cv2.imwrite(config.dataset_dir + f"/sample{ind:04d}-demo{i}.jpg", demo)


    def _img_to_feature_old(self, img):
        img = np.mean(img.astype(np.float32), axis=-1) # convert to gray scale

        mesh_grid = list(itertools.product(*[
                range(config.padding, config.img_size - config.padding - config.patch_size, config.stride),
            ] * 2))

        def make_patches(img):
            return np.stack([
                img[r:r+config.patch_size, c:c+config.patch_size] for r,c in mesh_grid
            ])

        return make_patches(img)

    def _reload(self):
        _data = ImagePreprocessor(base_dir=config.base_dataset_dir).get_dataset_full(data_selection='all', label_type='non-str')
        def _generate_raw_data():
            for label1, label2, imgs in _data:
                label = 0 in (label1 + label2)

                for img in imgs:
                    features = self.img_to_feature(img)
                    yield features, label

        gen = _generate_raw_data()
        self._pos_gen, self._neg_gen = branch_tee(gen, lambda val: val[1] is True)

    def _auto_reload(self, gen):
        while True:
            for val in gen: yield val
            self._reload()

    def pos_data_gen(self):
        return self._auto_reload(self._pos_gen)

    def neg_data_gen(self):
        return self._auto_reload(self._neg_gen)


import tensorflow as tf
def make_dataset():
    def feed_dataset(training):
        dg = DataGenerator()

        ds_type_arg = (
                (tf.float32, tf.int64),
                (tf.TensorShape([None, config.patch_size, config.patch_size]), tf.TensorShape([]))
            )

        ds_pos = tf.data.Dataset.from_generator(dg.pos_data_gen, *ds_type_arg)
        ds_neg = tf.data.Dataset.from_generator(dg.neg_data_gen, *ds_type_arg)
        ds = tf.data.Dataset.zip((ds_pos, ds_neg)).map(
                lambda d_pos, d_neg: (tf.stack((d_pos[0], d_neg[0])), tf.stack((d_pos[1], d_neg[1])))
                # d_pos: (feature1,`True`)
                # d_neg: (feature2,`False`)
                # @return: ([feature1, feature2], [`True`, `False`])
            )
        return ds

    return feed_dataset
