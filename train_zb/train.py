# -*- coding: utf-8 -*-
# @Author: gigaflw
# @Date:   2018-05-29 09:52:18
# @Last Modified by:   gigaflw
# @Last Modified time: 2018-06-01 16:26:05

import random
import numpy as np

random.seed(0)
np.random.seed(0)

import tensorflow as tf

from config import config
from dataset import make_dataset
from model import model


feed_dataset = make_dataset()
print("Dataset generator prepared")
classifier = tf.estimator.Estimator(
        model_fn=model,
        params={},
        model_dir=config.model_dir
    )

tf.logging.set_verbosity(tf.logging.INFO)
for epoch in range(1):
    classifier.train(
        input_fn=lambda : feed_dataset(training=True),
        steps=1000,
        hooks=[tf.train.LoggingTensorHook(tensors={'loss': 'loss'}, every_n_iter=1)]
    )
