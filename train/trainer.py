# @Author:      HgS_1217_
# @Create Date: 2018/5/28

import numpy
import tensorflow as tf
import numpy as np

from train.cnn import CNN
from config import DATASET_PATH
from data_process.image_preprocessor import ImagePreprocessor


TARGET_LABELS = [0, 1, 2, 3, 4, 5]
TEST_RATIO = 8 / 9


def generate_patch():
    """
    Generate patches in the dataset.
    :return:
    """
    image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
    image_pre.generate_patches()


def data_construction():
    """
    We separate the 500 data folders into 2 groups, 0-250 and 250-500. Since we have 270(180) photos
    each folder, we can cut the dataset into 540 batches. We set the train-test ratio to be 8/9, so
    train-set size is 480 and test-set size is 60.
    :return: raws, labels, test_raws, test_labels, loss_array
            loss_array: Weight loss matrix
    """
    image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
    l1, l2, d = image_pre.get_dataset_patched(size=20, data_selection='all', label_type='int', exist='new')

    r, l = [[] for _ in range(480)], [[] for _ in range(480)]
    tr, tl = [[] for _ in range(60)], [[] for _ in range(60)]
    label_counter = {target: 0 for target in TARGET_LABELS}

    for i in range(len(l1)):
        lbs = set(l1[i] + l2[i])
        sz = len(d[i])
        for target in TARGET_LABELS:
            if target in lbs:
                label_counter[target] += 1
        for j in range(sz):
            if j < int(sz * 8 / 9):
                base = 0 if i < int(len(l1) / 2) else 240
                r[j + base].append(d[i][j])
                label_array = []
                for target in TARGET_LABELS:
                    label_array.append([1, 0] if target in lbs else [0, 1])
                l[j + base].append(label_array)
            else:
                base = 0 if i < int(len(l1) / 2) else 30
                tr[j + base - int(sz * 8 / 9)].append(d[i][j])
                label_array = []
                for target in TARGET_LABELS:
                    label_array.append([1, 0] if target in lbs else [0, 1])
                tl[j + base - int(sz * 8 / 9)].append(label_array)

    raws, labels = [np.array(x) for x in r], [np.array(x) for x in l]
    test_raws, test_labels = [np.array(x) for x in tr], [np.array(x) for x in tl]
    loss_array = [[0.5 / (label_counter[label] / len(l1)), 0.5 / (1 - label_counter[label] / len(l1))]
                  for label in label_counter.keys()]

    print(raws[0].shape)
    print(labels[0].shape)
    print(loss_array)

    print("Dataset constructed")
    return raws, labels, test_raws, test_labels, loss_array


def train():
    raws, labels, test_raws, test_labels, loss_array = data_construction()

    # raws = raws[:50]
    # labels = labels[:50]
    # test_raws = test_raws[:10]
    # test_labels = test_labels[:10]

    cnn = CNN(raws, labels, test_raws, test_labels, epoch_size=100, loss_array=loss_array)
    cnn.train()

if __name__ == '__main__':
    # generate_patch()
    train()
