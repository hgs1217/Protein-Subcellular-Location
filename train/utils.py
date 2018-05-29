# @Author:      HgS_1217_
# @Create Date: 2018/5/28

import numpy as np


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    size, num_class = predictions.shape[0], predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    acc = [0 for _ in range(num_class)]
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc[ii] = 0.0
        else:
            acc[ii] = np.diag(hist)[ii] / float(hist.sum(1)[ii])
    return np.nanmean(acc_total), np.nanmean(iu), acc
