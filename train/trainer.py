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

    r, l, tr, tl = [[] for _ in range(480)], [[] for _ in range(480)], \
                   [[] for _ in range(60)], [[] for _ in range(60)]

    target_label = 1
    cnt = 0
    for i in range(len(l1)):
        lbs = set(l1[i] + l2[i])
        sz = len(d[i])
        if target_label in lbs:
            cnt += 1
        for j in range(sz):
            if j < int(sz * 8 / 9):
                base = 0 if i < int(len(l1) / 2) else 240
                r[j + base].append(d[i][j])
                l[j + base].append([1, 0] if target_label in lbs else [0, 1])
            else:
                base = 0 if i < int(len(l1) / 2) else 30
                tr[j + base - int(sz * 8 / 9)].append(d[i][j])
                tl[j + base - int(sz * 8 / 9)].append([1, 0] if target_label in lbs else [0, 1])

    for i in range(480):
        raws.append(np.mean(np.array(r[i]), axis=3, keepdims=True))
        labels.append(np.array(l[i]))
        if i < 60:
            test_raws.append(np.mean(np.array(tr[i]), axis=3, keepdims=True))
            test_labels.append(np.array(tl[i]))

    print(raws[0].shape)
    print(labels[0].shape)

    # raws = raws[:50]
    # labels = labels[:50]
    # test_raws = test_raws[:10]
    # test_labels = test_labels[:10]

    # for i in range(len(l1)):
    #     lbs = set(l1[i] + l2[i])
    #     sz = len(d[i])
    #     raws.append(np.mean(np.array(d[i][:int(sz*8/9)]), axis=3, keepdims=True))
    #     test_raws.append(np.mean(np.array(d[i][int(sz*8/9):]), axis=3, keepdims=True))
    #     labels.append(np.array([[1 if 0 in lbs else 0] for _ in range(int(sz*8/9))]))
    #     test_labels.append(np.array([[1 if 0 in lbs else 0] for _ in range(int(sz*1/9))]))

    print(cnt)
    print([0.5 / (cnt / len(l1)), 0.5 / (1 - cnt / len(l1))])
    print("Dataset constructed")
    vgg16 = VGG16(raws, labels, test_raws, test_labels, batch_size=train_size, epoch_size=100, classes=2,
                  loss_array=[0.5 / (cnt / len(l1)), 0.5 / (1 - cnt / len(l1))])
    vgg16.train()

if __name__ == '__main__':
    # generate_patch()
    train()
