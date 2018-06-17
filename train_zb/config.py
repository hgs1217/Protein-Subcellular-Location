# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-06-01 16:25:21
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-17 22:53:28

import os

class config:
    model_dir = None
    base_dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'data_process', 'HPA_ieee')
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')

    # dataset config
    stride = 128
    patch_size = 32
    img_size = 3000
    padding = 750

    # training config
    shuffle_size = 10
    learning_rate = 0.1

