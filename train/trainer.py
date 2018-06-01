# @Author:      HgS_1217_
# @Create Date: 2018/5/28

from train.cnn import CNN
from data_process.data_preprocessor import generate_patch, data_construction


TARGET_LABELS = [0, 1, 2, 3, 4, 5]
TEST_RATIO = 8 / 9


def train():
    raws, labels, test_raws, test_labels, loss_array = data_construction(TARGET_LABELS, TEST_RATIO)

    print(raws[0].shape)
    print(labels[0].shape)
    print(loss_array)

    # raws = raws[:50]
    # labels = labels[:50]
    # test_raws = test_raws[:10]
    # test_labels = test_labels[:10]

    cnn = CNN(raws, labels, test_raws, test_labels, epoch_size=100, loss_array=loss_array)
    cnn.train()

if __name__ == '__main__':
    # generate_patch()
    train()
