# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from train_tth.config import DATA_DIR


def data_construction(classifier_num, ratio=8 / 9):
    labels = []
    cv_data = []

    classifier_dir = os.path.join(DATA_DIR, 'classfier_' + classifier_num)
    group_set = os.listdir(classifier_dir)
    for item in group_set:
        patches_dir = os.path.join(classifier_dir, item)
        patch_set = []
        patches = os.listdir(patches_dir)
        os.chdir(patches_dir)
        for a_patch in patches:
            cv_image = cv2.imread(a_patch)
            patch_set.append(cv_image)
        txt_address = os.path.join(patches_dir, 'output.txt')
        with open(txt_address, 'r') as f:
            cur_label = []
            i = 0
            for j in f.readline():
                cur_label.append(float(j))
                i += 1

        labels.append(cur_label)
        cv_data.append(patch_set)
    r, l = [[] for _ in range(240)], [[] for _ in range(240)]
    tr, tl = [[] for _ in range(30)], [[] for _ in range(30)]
    # label_counter = {target: 0 for target in target_labels}

    for i in range(len(labels)):

        sz = len(cv_data[i])

        for j in range(sz):
            if j < int(sz * ratio):

                r[j].append(cv_data[i][j])

                l[j].append(labels[i])
            else:

                tr[j - int(sz * ratio)].append(cv_data[i][j])

                tl[j - int(sz * ratio)].append(labels[i])
    raws, labels = [np.array(x) for x in r], [np.array(x) for x in l]
    test_raws, test_labels = [np.array(x) for x in tr], [np.array(x) for x in tl]

    return raws, labels, test_raws, test_labels
