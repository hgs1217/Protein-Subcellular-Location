# @Author:      HgS_1217_
# @Create Date: 2018/5/28

import numpy

from trainer.vgg16 import VGG16
from config import DATASET_PATH
from data_process.image_preprocessor import ImagePreprocessor


class Trainer(object):

    def __init__(self, data_dir):
        self._image_pre = ImagePreprocessor(base_dir=data_dir)
        self._vgg16 = VGG16()



def train():
    image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
    # vgg16 = VGG16()
    for idx, (l1, l2, d) in enumerate(image_pre.get_dataset_patched(size=20, data_selection='all')):
        # print("Dealing with {0}".format(idx))
        # label_list = l1.split(";") + l2.split(";")
        # if "Cytoplasm" in label_list:
        #     print(idx)
        print(l1, l2)
        break
    for idx, (l1, l2, d) in enumerate(image_pre.get_dataset_patched(size=20, data_selection='all')):
        # print("Dealing with {0}".format(idx))
        # label_list = l1.split(";") + l2.split(";")
        # if "Cytoplasm" in label_list:
        #     print(idx)
        print(l1, l2)
        break


if __name__ == '__main__':
    train()
