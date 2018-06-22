# -*- coding: utf-8 -*-
from train_tth.train import train
from train_tth.data_split import data_slice

if __name__ == '__main__':
    # train(start_step=89, epoch_size=11, detail_log=True)
    # train(start_step=0, epoch_size=100, detail_log=True, new_ckpt_internal=-1, network_mode="", simple=True)
    data_slice()
    for i in range(6):
        train(classfier_num=i,start_step=0, epoch_size=100, detail_log=True, new_ckpt_internal=-1,  simple=False, gpu=False)
