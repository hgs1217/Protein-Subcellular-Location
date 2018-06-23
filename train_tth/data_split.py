# -*- coding: utf-8 -*-
import cv2
import os
import argparse
import numpy
import xlrd
import shutil
from train_tth.config import RAW_DATA_DIR, LABELS, TARGET_LABELS


class Sorted_Tree:
    def __init__(self):
        self.__tree_structure_down_top = {1: 14, 0: 16, 2: 16, 3: 13, 4: 15, 5: 15, 15: 13, 13: 12, 12: 11, 10: 11,
                                          16: 14, 14: 12}
        self.__tree_structure_top_down = {11: [12, 10], 12: [13, 14], 13: [15, 3], 15: [4, 5], 14: [16, 1], 16: [2, 0]}

    def generate_count(self, label_count):
        result = [0 for i in range(17)]
        for target in label_count:
            if label_count[target] != 0:
                result[target] += 1
                while self.__tree_structure_down_top[target] != 11:
                    target = self.__tree_structure_down_top[target]
                    result[target] += 1
        output = []
        for i in range(11, 17):
            left = result[self.__tree_structure_top_down[i][0]]
            right = result[self.__tree_structure_top_down[i][1]]
            if left == 0 and right == 0:
                output.append([-1, -1])
            else:
                output.append([left / (left + right), right / (left + right)])
        return output


def data_slice():
    xlsx_path = os.path.join(RAW_DATA_DIR, 'data_label.xlsx')

    output_data_dir = os.path.join(RAW_DATA_DIR, 'sliced_data')
    if 'sliced_data' not in os.listdir(RAW_DATA_DIR):
        os.mkdir(output_data_dir)
    os.chdir(output_data_dir)
    if 'classfier_0' not in os.listdir(output_data_dir):
        for i in range(6):
            os.mkdir('classfier_' + str(i))

    # print(xlsx_path)
    book_sheet = xlrd.open_workbook(xlsx_path).sheet_by_index(0)
    # print (book_sheet)
    sets = book_sheet.col_values(0)
    pos1 = book_sheet.col_values(6)
    pos2 = book_sheet.col_values(7)
    reli = book_sheet.col_values(8)
    if sets[0] == '基因编号':
        sets = sets[1:]
        pos1 = pos1[1:]
        pos2 = pos2[1:]
        reli = reli[1:]
    for i in range(len(pos1)):
        # 这里有个很奇怪的bug，直接去头尾一个字符会导致有的字符串第一个字没了，所以写成这样
        pos1[i] = pos1[i].replace('\'', '').replace('[', '').replace(']', '')
        pos2[i] = pos2[i].replace('\'', '').replace('[', '').replace(']', '')
        sets[i] = sets[i].replace('\'', '').replace('[', '').replace(']', '')
        reli[i] = reli[i][1:-1]

    raw_data = os.path.join(RAW_DATA_DIR, 'data')
    dir_sets = os.listdir(raw_data)
    dir_sets = [a_set for a_set in dir_sets if a_set.startswith('liver_ENSG')]
    print(len(dir_sets))
    st = Sorted_Tree()
    cnt = 0
    for item in dir_sets:
        ensg_index = item.split('_')[1]
        pos_index = sets.index(ensg_index)
        # print (pos_index)
        label1 = [LABELS[x] for x in pos1[pos_index].split(';')]
        label2 = [LABELS[x] for x in pos2[pos_index].split(';')]
        labels = set(label1 + label2)
        # print (labels)

        label_counter = {target: 0 for target in TARGET_LABELS}

        if len(labels) == 1 and 10 in labels:
            label_counter[10] = 1
            # cnt+=1

        else:
            for target in TARGET_LABELS:
                if target in labels:
                    label_counter[target] += 1
            label_counter[10] = 0
        classifier_sets = st.generate_count(label_count=label_counter)
        for i in range(len(classifier_sets)):
            if classifier_sets[i][0] != -1:
                postfix = 'classfier_' + str(i)
                dst_dir = os.path.join(output_data_dir, postfix, item)
                src_dir = os.path.join(raw_data, item, 'new')
                shutil.copytree(src_dir, dst_dir)
                cur_output_file = os.path.join(dst_dir, 'output.txt')
                with  open(cur_output_file, 'w') as f:
                    for j in classifier_sets[i]:
                        f.write(str(j) + '\n')
            else:
                if i == 0:
                    print(labels)
    # print (cnt)
    return 0


if __name__ == '__main__':
    data_slice()
