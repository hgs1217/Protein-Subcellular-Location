# @Author:      HgS_1217_
# @Create Date: 2018/5/28

from train.vgg16 import VGG16
from data_process.image_preprocessor import ImagePreprocessor
from config import DATASET_PATH


def train():
    image_pre = ImagePreprocessor(base_dir=DATASET_PATH)
    # vgg16 = VGG16()
    # for l1, l2, d in image_pre.get_dataset_patched(size=20, data_selection='all'):

if __name__ == '__main__':
    train()
