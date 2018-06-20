# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:56:34
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-20 13:30:02

import os, re, pickle, time, itertools, random
import numpy as np

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
    def __init__(self, sample_range=[0, -1], shuffle_samples=False, max_patches_per_sample=-1, dataset='dumped'):
        self._dataset = dataset

        self._sample_slice = slice(*sample_range)
        self._max_patches = max_patches_per_sample
        self._shuffle_samples = shuffle_samples

        self._current_label_index = 0   # 0 ~ 5
        self._reload()

    def _reload(self):
        if self._dataset == 'dumped':
            self._reload_from_dumped()
        else:
            self._reload_from_images()

    def _reload_from_dumped(self):
        sample_files = sorted(f for f in os.listdir(config.dataset_dir) if re.match(r'sample(\d+)_([01]+)', f))
        sample_files = sample_files[self._sample_slice]

        if len(sample_files) == 0:
            raise ValueError(f"sample index range {self._sample_slice} out of range")

        if self._shuffle_samples:
            random.shuffle(sample_files)

        sample_files = [ (f, [x == '1' for x in re.match(r'sample\d+_([01]+)', f)[1]]) for f in sample_files]
        def _generate_raw_data(filter_label=None):
            for filename, labels in sample_files:
                # labels is an onehot like [False, True, False, False, ...]
                if filter_label and not filter_label(labels):
                    continue

                data = pickle.load(open(config.dataset_dir/filename, 'br'))
                patches = data['patches']
                if len(patches) < 1000:
                    continue

                if self._max_patches > 0:
                    _choice = np.zeros(len(patches), dtype=np.bool)
                    _choice[:self._max_patches] = True
                    np.random.shuffle(_choice)
                    patches = patches[_choice]
                patches /= 255.0

                # label_index = random.randint(0, config.n_labels-1)
                print(f"{filename} {len(data['patches'])} -> {len(patches)} patches loaded")

                labels = [int(l) for l in labels]
                yield patches, labels

        self._pos_gen = _generate_raw_data(lambda label: label[self._current_label_index] is True)
        self._neg_gen = _generate_raw_data(lambda label: label[self._current_label_index] is False)
        self._gen = _generate_raw_data()

    def _auto_reload(self, gen):
        while True:
            for val in gen: yield val
            self._reload()

    def pos_data_gen(self):
        return self._auto_reload(self._pos_gen)

    def neg_data_gen(self):
        return self._auto_reload(self._neg_gen)

    def pos_neg_pair_data_gen(self):
        for (pos_patches, pos_labels), (neg_patches, neg_labels) in zip(self.pos_data_gen(), self.neg_data_gen()):
            yield pos_patches, neg_patches, self._current_label_index

            self._current_label_index = random.randint(0, 5)
            print(f"label idnex changed to {self._current_label_index}")

    def data_gen(self):
        return self._auto_reload(self._gen)

    @staticmethod
    def dump(patch_size, stride, cut_img_threshold, expected_max_patches_per_img):
        import cv2

        _data = ImagePreprocessor(base_dir=config.base_dataset_dir).get_dataset_full(data_selection='sup', label_type='non-str')
        log_file = open(config.dataset_dir/"log.txt", 'w')
        tic = time.time()
        print(f"Config: patch_size={patch_size}x{patch_size} stride={stride}x{stride} threshold={cut_img_threshold}", file=log_file)

        def get_sample(imgs):
            patches, positions, demos = zip(*[cut_img(img,
                width=patch_size, height=patch_size,
                xstep=stride, ystep=stride,
                threshold=cut_img_threshold,
                write_mode=False, output_folder=None
            ) for img in imgs])
            patches = np.vstack([np.stack(p) for p in patches if len(p) > 0])

            # dropout excessive patches to keep the size of dateset reasonable
            rate = 1 - np.exp(-expected_max_patches_per_img / len(patches))
            random_choice = np.random.random(len(patches)) < rate
            patches = patches[random_choice]
            patches = np.mean(patches.astype(np.float32), axis=-1)   # convert to gray scale
            return patches, demos

        for ind, (label1, label2, imgs) in enumerate(_data):
            labels = set(l for l in label1 + label2 if l < config.n_labels)
            patches, demos = get_sample(imgs)

            with open(config.dataset_dir/f"sample{ind:04d}", 'bw') as f:
                pickle.dump({'patches': patches, 'labels': labels}, f)

            for i, demo in enumerate(demos):
                cv2.imwrite(config.dataset_dir/"demo"/f"sample{ind:04d}-{i}.jpg", demo)

            print(f"Dumping sample{ind:04d} with patches: {patches.shape}, labels: {labels}", file=log_file)
            if ind % 10 == 9:
                toc = time.time()
                print(f"Speed: {toc - tic:.3f} sec / {ind + 1} samples = {(toc - tic) / (ind + 1):.3f} sec/samples", file=log_file)
            log_file.flush()
        tac = time.time()
        log_file.close()

    # def _reload_from_images(self):
    #     _data = ImagePreprocessor(base_dir=config.base_dataset_dir).get_dataset_full(data_selection='all', label_type='non-str')
    #     def _generate_raw_data():
    #         for label1, label2, imgs in _data:
    #             label = 0 in (label1 + label2)

    #             for img in imgs:
    #                 features = self.img_to_feature(img)
    #                 yield features, label

    #     gen = _generate_raw_data()
    #     self._pos_gen, self._neg_gen = branch_tee(gen, lambda val: val[1] is True)
    #     self._gen = _generate_raw_data()

    # def img_to_feature(self, img):
    #     # img: width x height x 3
    #     patches, positions, demo = cut_img(img)
    #     patches, positions = map(np.stack, (patches, positions))
    #     patches = np.mean(patches.astype(np.float32), axis=-1)   # convert to gray scale
    #     # patches: N x width x height
    #     # positions: N x 2
    #     return patches

    # def _img_to_feature_old(self, img):
    #     img = np.mean(img.astype(np.float32), axis=-1) # convert to gray scale

    #     mesh_grid = list(itertools.product(*[
    #             range(config.padding, config.img_size - config.padding - config.patch_size, config.stride),
    #         ] * 2))

    #     def make_patches(img):
    #         return np.stack([
    #             img[r:r+config.patch_size, c:c+config.patch_size] for r,c in mesh_grid
    #         ])

    #     return make_patches(img)

    def make_dataset(self, split_pos_neg):
        import tensorflow as tf

        if split_pos_neg:
            ds_type_arg = (
                    (tf.float32, tf.float32, tf.int64),
                    (
                        tf.TensorShape([None, config.patch_size, config.patch_size]),
                        tf.TensorShape([None, config.patch_size, config.patch_size]),
                        tf.TensorShape([])
                    )
                )

            ds = tf.data.Dataset.from_generator(self.pos_neg_pair_data_gen, *ds_type_arg)
        else:
            ds_type_arg = (
                    (tf.float32, tf.int64),
                    (tf.TensorShape([None, config.patch_size, config.patch_size]), tf.TensorShape([6, ]))
                )
            ds = tf.data.Dataset.from_generator(self.data_gen, *ds_type_arg)
        return ds

if __name__ == '__main__':
    pass
    # print(make_dataset()(True))
    # DataGenerator().dump(
    #     patch_size=32,
    #     stride=16,
    #     cut_img_threshold=0.9,
    #     expected_max_patches_per_img=12000
    # )
