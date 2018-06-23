# @Author:      HgS_1217_
# @Create Date: 2018/6/1

import numpy as np

from config import DATASET_PATH
from data_process.image_preprocessor import ImagePreprocessor


def generate_patch(dataset_path=DATASET_PATH):
    """
        Generate patches in the dataset.
        :return:
    """
    image_pre = ImagePreprocessor(base_dir=dataset_path)
    image_pre.generate_patches()


def data_construction(target_labels, ratio=8 / 9):
    """
        We separate the 500 data folders into 2 groups, 0-250 and 250-500. Since we have 270(180) photos
        each folder, we can cut the dataset into 540 batches. Take the default train-test ratio 8/9 as an
        examle. The train-set size is 480 and test-set size is 60.

        :param target_labels: list, [label number]
                            The labels needed in the training.
        :param ratio: float, default 0.8888888
                    The ratio of training set over test set.
        :return: raws, labels, test_raws, test_labels, loss_array
                raws: list, [training set size * photo numbers * photo width * photo height * channels],
                    default [480 * 250 * 20 * 20 * 3]
                    Training set data.
                labels: list, [training set size * photo numbers * label number * 2], default
                    [480 * 250 * 6 * 2]
                    Training set labels. The third dimension is the one hot encoding, so that [1, 0] means
                    inclusion and [0, 1] means exclusion.
                test_raws: list, [test set size * photo numbers * photo width * photo height * channels],
                    default [60 * 250 * 20 * 20 * 3]
                    Test set data.
                test_labels: list, [test set size * photo numbers * label number * 2], default [60 * 250 * 6 * 2]
                    Test set labels. The shape is the same to @labels.
                loss_array: list, [label number * 2], default [6 * 2]
                    This is the weight loss matrix, which is used in weighted loss function. The formulation
                    of loss array is:
                        [0.5 / (label counter / label size), 0.5 / (1 - label counter / label size)]
                    Loss array provide a balance in training for data under non-uniform distribution.
    """
    image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
    l1, l2, d = image_pre.get_dataset_patched(size=20, data_selection='all', label_type='int', exist='new')

    r, l = [[] for _ in range(480)], [[] for _ in range(480)]
    tr, tl = [[] for _ in range(60)], [[] for _ in range(60)]
    label_counter = {target: 0 for target in target_labels}

    for i in range(len(l1)):
        lbs = set(l1[i] + l2[i])
        sz = len(d[i])
        for target in target_labels:
            if target in lbs:
                label_counter[target] += 1
        for j in range(sz):
            if j < int(sz * ratio):
                base = 0 if i < int(len(l1) / 2) else 240
                r[j + base].append(d[i][j])
                label_array = []
                for target in target_labels:
                    label_array.append([1, 0] if target in lbs else [0, 1])
                l[j + base].append(label_array)
            else:
                base = 0 if i < int(len(l1) / 2) else 30
                tr[j + base - int(sz * ratio)].append(d[i][j])
                label_array = []
                for target in target_labels:
                    label_array.append([1, 0] if target in lbs else [0, 1])
                tl[j + base - int(sz * ratio)].append(label_array)

    raws, labels = [np.array(x) for x in r], [np.array(x) for x in l]
    test_raws, test_labels = [np.array(x) for x in tr], [np.array(x) for x in tl]
    loss_array = [[0.5 / (label_counter[label] / len(l1)), 0.5 / (1 - label_counter[label] / len(l1))]
                  for label in label_counter.keys()]

    print("Dataset constructed")
    return raws, labels, test_raws, test_labels, loss_array


def data_construction_v2(target_labels, batch_size, ratio=8 / 9):
    image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
    l1, l2, d = image_pre.get_dataset_patched(size=20, data_selection='all', label_type='int', exist='new')
    batch_num = int(500 * 270 / batch_size)  # batch_num=5400 if batch_size=25

    r, l = [[] for _ in range(int(batch_num * 8 / 9))], [[] for _ in range(int(batch_num * 8 / 9))]
    tr, tl = [[] for _ in range(int(batch_num / 9))], [[] for _ in range(int(batch_num / 9))]
    label_counter = {target: 0 for target in target_labels}

    for i in range(len(l1)):
        lbs = set(l1[i] + l2[i])
        sz = len(d[i])
        for target in target_labels:
            if target in lbs:
                label_counter[target] += 1
        for j in range(sz):
            if j < int(sz * ratio):
                base = 0 if i < int(len(l1) / 2) else int(batch_num * 4 / 9)
                r[j + base].append(d[i][j])
                label_array = []
                for target in target_labels:
                    label_array.append([1, 0] if target in lbs else [0, 1])
                l[j + base].append(label_array)
            else:
                base = 0 if i < int(len(l1) / 2) else int(batch_num / 18)
                tr[j + base - int(sz * ratio)].append(d[i][j])
                label_array = []
                for target in target_labels:
                    label_array.append([1, 0] if target in lbs else [0, 1])
                tl[j + base - int(sz * ratio)].append(label_array)

    raws, labels = [np.array(x) for x in r], [np.array(x) for x in l]
    test_raws, test_labels = [np.array(x) for x in tr], [np.array(x) for x in tl]
    loss_array = [[0.5 / (label_counter[label] / len(l1)), 0.5 / (1 - label_counter[label] / len(l1))]
                  for label in label_counter.keys()]

    print("Dataset constructed with batch size {}".format(batch_size))
    return raws, labels, test_raws, test_labels, loss_array
