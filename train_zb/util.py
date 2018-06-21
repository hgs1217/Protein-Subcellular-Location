# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-06-21 13:40:51
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-21 14:37:20

import os
class Path(str):
    def __init__(self, path):
        self.path = path

    def __truediv__(self, path):
        return Path(os.path.join(self.path, str(path)))

    def __rtruediv__(self, path):
        return Path(os.path.join(str(path), self.path))


def float_list_to_str(lst, format_str='{:.2f}'):
    ret = ','.join([format_str.format(x) for x in lst])
    return '[' + ret  + ']'

import numpy as np
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

import time
class Timer:
    def __init__(self):
        self.t_start = self.t_end = 0
        self.rounds = 1

    def tic(self):
        self.t_start = time.time()
        
    def toc(self):
        self.t_end = time.time()

    @property
    def time_sec(self):
        return (self.t_end - self.t_start) / self.rounds

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.toc()

    def __call__(self, rounds=1):
        self.rounds = rounds
        return self
