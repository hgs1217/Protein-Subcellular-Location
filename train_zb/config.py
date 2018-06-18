# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-06-01 16:25:21
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-18 15:42:51

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

    # dataset config
    stride = 16
    patch_size = 32
    img_size = 3000
    cut_img_threshold = 0.5
    padding = 750

    # training config
    shuffle_size = 10
    learning_rate = 0.1
