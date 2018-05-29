# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:52:18
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-05-29 12:29:23

import random
import numpy as np

random.seed(0)
np.random.seed(0)

import tensorflow as tf

from dataset import make_dataset, get_data
from model import model

class config:
    model_dir = None
    batch_size = 1

feed_dataset = make_dataset()
classifier = tf.estimator.Estimator(
        model_fn=model,
        params={},
        model_dir=config.model_dir
    )

tf.logging.set_verbosity(tf.logging.INFO)
for epoch in range(1):
    classifier.train(
        input_fn=lambda : feed_dataset(config.batch_size, training=True),
        steps=100,
        # hooks=[tf.train.LoggingTensorHook(tensors={}, every_n_iter=100)]
    )
