# -*- coding: utf-8 -*-
import os
import shutil
import math

import random
import time
import numpy as np
import tensorflow as tf
from train.utils import per_class_acc, variable_with_weight_decay, add_loss_summaries, \
    conv_layer, avg_pool_layer, fc_layer, norm_layer, weighted_loss

# path1 = "/Users/tangtonghui/Downloads/test"
# path2 = "/Users/tangtonghui/Downloads/test111"
# shutil.copytree(path1,path2)
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
test_total_data = np.zeros((10, 2))
test_total_data[5][0]=0.5
test_total_data[8][0]=0.7
print (test_total_data)
print (  np.mean(test_total_data, axis=0))