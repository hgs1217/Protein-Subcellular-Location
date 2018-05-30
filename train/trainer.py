# @Author:      HgS_1217_
# @Create Date: 2018/5/28

import numpy
import tensorflow as tf
import numpy as np

from train.vgg16 import VGG16
from config import DATASET_PATH
from data_process.image_preprocessor import ImagePreprocessor


def generate_patch():
    image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
    image_pre.generate_patches()


def train():
    image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
    l1, l2, d = image_pre.get_dataset_patched(size=20, data_selection='all', label_type='int', exist='new')
    train_size, test_size = 240, 30

    raws, labels, test_raws, test_labels = [], [], [], []

    for i in range(len(l1)):
        lbs = set(l1[i] + l2[i])
        sz = len(d[i])
        raws.append(np.mean(np.array(d[i][:int(sz*8/9)]), axis=3, keepdims=True))
        test_raws.append(np.mean(np.array(d[i][int(sz*8/9):]), axis=3, keepdims=True))
        labels.append(np.array([[1 if 0 in lbs else 0] for _ in range(int(sz*8/9))]))
        test_labels.append(np.array([[1 if 0 in lbs else 0] for _ in range(int(sz*1/9))]))

    vgg16 = VGG16(raws, labels, test_raws, test_labels, batch_size=train_size, epoch_size=100, classes=1)
    vgg16.train()

if __name__ == '__main__':
    # generate_patch()
    train()
