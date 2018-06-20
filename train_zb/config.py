# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-06-01 16:25:21
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-20 15:25:36

import os

class Path(str):
    def __init__(self, path):
        self.path = path

    def __truediv__(self, path):
        return Path(os.path.join(self.path, str(path)))

    def __rtruediv__(self, path):
        return Path(os.path.join(str(path), self.path))

class config:
    model_dir = None
    base_dataset_dir = Path(os.path.dirname(__file__))/'..'/'data_process'/'HPA_ieee'
    dataset_dir = Path(os.path.dirname(__file__))/'dataset'
    ckpt_dir = Path(os.path.dirname(__file__))/'ckpt'
    summary_dir = Path(os.path.dirname(__file__))/'summary'
    n_labels = 6

    # dataset config
    stride = 16
    patch_size = 32
    img_size = 3000
    cut_img_threshold = 0.9
    padding = 750
    train_samples = [0, 200]
    eval_samples = [200, 230]
    # batch_size = 1024

    # training config
    # shuffle_size = batch_size * 3
    learning_rate = 0.1
    max_patches_per_sample = 2048
    n_candidates = int(max_patches_per_sample * 0.2)