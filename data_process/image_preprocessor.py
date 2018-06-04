# -*- coding: utf-8 -*-  
"""
@author: Suibin Sun
@file: image_preprocessor.py
@time: 2018/5/9 16:17

Usage:
    $ python image_preprocessor.py [-h] [-b BASE_DIR] [-w WIDTH] [-f FOLDER_NAME]

"""
import cv2
import os
import argparse
import numpy
import xlrd


class ImagePreprocessor(object):
    """
    Generate slices of images with given center point.

    Attributes:
        base_dir (str): address of <HPA_ieee> directory，e.g.：X:/****/HPA_ieee
        width (int): width(for type square) or diameter(for type circle) of each slice
        new_dir_name (str): name of new directory, e.g.: 'framed'. New folder will be in 'data/liver_XXX/<new_folder>'.
                            Just ignore this feature!

    """

    def __init__(self, width=20, new_dir_name='new', base_dir=os.getcwd()):
        self.__base_dir = base_dir
        self.__data_dir = os.path.join(base_dir, 'data')
        self.__width = int(int(width) / 2)
        self.__new_dir_name = new_dir_name
        self.__labels = {
            'Cytoplasm': 0,
            'Mitochondria': 1,
            'Nucleus but not nucleoli': 2,
            'Nucleus': 2,
            'Nucleoli': 2,
            'Nuclear membrane': 2,
            'Vesicles': 3,
            'Endoplasmic reticulum': 4,
            'Golgi apparatus': 5,
            'Cytoskeleton (Intermediate filaments)': 6,
            'Cytoskeleton (Microtubules)': 6,
            'Cytoskeleton (Actin filaments)': 6,
            'Microtubule organizing center': 7,
            'Centrosome': 7,
            'Plasma membrane': 8,
            'Focal Adhesions': 9,
            'Cell Junctions': 9,
            '': 10
        }

    def __generate_slices_a_set(self, set_id):
        """
        generate patched for a sample data and save them in a new folder
        :param set_id: folder name of an data instance, like 'liver_ENSG00000004975_22914'
        :return:
        """
        image_set_path = os.path.join(self.__data_dir, set_id)
        os.chdir(image_set_path)
        try:
            os.mkdir(self.__new_dir_name)
        except FileExistsError:
            print('Directory already exist! Pass making new directory.')
        slice_dir = os.path.join(image_set_path, self.__new_dir_name)
        image_dir = os.path.join(image_set_path, 'normal')
        texts = os.listdir(image_dir)
        os.chdir(image_dir)
        for text in texts:
            if text.endswith('txt'):
                image_name = text.split('.')[0]
                # image_path = os.path.join(image_dir, '{}.jpg'.format(image_name))
                points = []
                count = 0
                with open(text, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        points.append(line.split(' '))
                cv_image = cv2.imread("{}.jpg".format(image_name))
                for point in points:
                    count += 1
                    x = int(point[0])
                    y = int(point[1])
                    a_slice = cv_image[
                              x - self.__width:
                              x + self.__width,
                              y - self.__width:
                              y + self.__width]
                    new_dir = os.path.join(slice_dir, '{0}_p{1}.jpg'.format(image_name, count))
                    cv2.imwrite(new_dir, a_slice)

    def generate_patches(self):
        """
        Used to create patched image files and save them in different folder
        :return:
        """
        sets = os.listdir(self.__data_dir)
        sets = [a_set for a_set in sets if a_set.startswith('liver_ENSG')]
        total = len(sets)
        count = 1
        for a_set in sets:
            print('\r', 'slicing {0}/{1}'.format(count, total).ljust(20), end='')
            self.__generate_slices_a_set(a_set)
            count += 1

    def __get_full_a_set(self, set_id):
        """
        :param set_id: folder name of an data instance, like 'liver_ENSG00000004975_22914'
        :return: a list of cv_data
        """
        image_dir = os.path.join(self.__data_dir, set_id, 'normal')
        texts = os.listdir(image_dir)
        os.chdir(image_dir)
        full_set = []
        for text in texts:
            if text.endswith('jpg'):
                cv_image = cv2.imread(text)
                full_set.append(cv_image)
        return full_set

    def get_dataset_full(self, data_selection='all', label_type='str'):
        """
        :param label_type: 'str' or other
        :param data_selection: must be in ['all', 'sup', 'usable']
                all -> all data
                sup -> data with supportive labels
                usable -> data with nonempty labels
        :return: label1, label2, cv_data
                label string is something like <Nucleus;Cytoplasm;Golgi apparatus> WITHOUT quote
                cv_data is in the shape of (3*3000*3000*3)
        """
        dir_sets = os.listdir(self.__data_dir)
        dir_sets = [a_set for a_set in dir_sets if a_set.startswith('liver_ENSG')]

        xlsx_path = os.path.join(self.__base_dir, 'data_label.xlsx')
        book_sheet = xlrd.open_workbook(xlsx_path).sheet_by_index(0)
        sets = book_sheet.col_values(0)
        pos1 = book_sheet.col_values(6)
        pos2 = book_sheet.col_values(7)
        reli = book_sheet.col_values(8)
        if sets[0] == '基因编号':
            sets = sets[1:]
            pos1 = pos1[1:]
            pos2 = pos2[1:]
            reli = reli[1:]

        # remove <'> at head and tail, while "[]" becomes ""(empty string)
        for i in range(len(pos1)):
            pos1[i] = pos1[i].replace('\'', '').replace('[', '').replace(']', '')
            pos2[i] = pos2[i].replace('\'', '').replace('[', '').replace(']', '')
            sets[i] = sets[i].replace('\'', '').replace('[', '').replace(']', '')
            reli[i] = reli[i][1:-1]
        if data_selection == 'all':
            for a_set in dir_sets:
                ensg_index = a_set.split('_')[1]
                a_full_set = self.__get_full_a_set(a_set)
                pos_index = sets.index(ensg_index)
                if label_type == 'str':
                    yield pos1[pos_index].split(';'), pos2[pos_index].split(';'), a_full_set
                else:
                    label1 = [self.__labels[x] for x in pos1[pos_index].split(';')]
                    label2 = [self.__labels[x] for x in pos2[pos_index].split(';')]
                    yield label1, label2, a_full_set
        elif data_selection == 'sup':
            suffices = book_sheet.col_values(2)[1:]
            for i in range(len(reli)):
                if reli[i] == 'Supportive':
                    a_full_set = self.__get_full_a_set('liver_{}_{}'.format(sets[i], suffices[i][1:-1]))
                    if label_type == 'str':
                        yield pos1[i].split(';'), pos2[i].split(';'), a_full_set
                    else:
                        label1 = [self.__labels[x] for x in pos1[i].split(';')]
                        label2 = [self.__labels[x] for x in pos2[i].split(';')]
                        yield label1, label2, a_full_set
        elif data_selection == 'usable':
            suffices = book_sheet.col_values(2)[1:]
            for i in range(len(reli)):
                if reli[i] != '':
                    a_full_set = self.__get_full_a_set('liver_{}_{}'.format(sets[i], suffices[i][1:-1]))
                    if label_type == 'str':
                        yield pos1[i].split(';'), pos2[i].split(';'), a_full_set
                    else:
                        label1 = [self.__labels[x] for x in pos1[i].split(';')]
                        label2 = [self.__labels[x] for x in pos2[i].split(';')]
                        yield label1, label2, a_full_set

    def __set_width(self, width):
        """
        temporally initialize self.__width
        :param width: the width of patch image
        :return:
        """
        self.__width = width

    def __get_patch_a_set(self, set_id):
        """
        :param set_id: folder name of an data instance, like 'liver_ENSG00000004975_22914'
        :return: a list of cv_data, the shape is (270*size*size*3)
        """
        patch_set = []
        image_dir = os.path.join(self.__data_dir, set_id, 'normal')
        texts = os.listdir(image_dir)
        os.chdir(image_dir)
        for text in texts:
            if text.endswith('txt'):
                image_name = text.split('.')[0]
                points = []
                with open(text, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        points.append(line.split(' '))
                cv_image = cv2.imread("{}.jpg".format(image_name))
                for point in points:
                    x = int(point[0])
                    y = int(point[1])
                    a_patch = cv_image[
                              x - self.__width:
                              x + self.__width,
                              y - self.__width:
                              y + self.__width]
                    patch_set.append(a_patch)
        return patch_set

    def __get_generated_patch_a_set(self, set_id, new_dir_name):
        """
        :param set_id:
        :param new_dir_name:
        :return:
        """
        patch_set = []
        image_dir = os.path.join(self.__data_dir, set_id, new_dir_name)
        patches = os.listdir(image_dir)
        os.chdir(image_dir)
        for a_patch in patches:
            cv_image = cv2.imread(a_patch)
            patch_set.append(cv_image)
        return patch_set

    def get_dataset_patched(self, size=20, data_selection='all', label_type='str', exist=None):
        """
        :param exist: None or a string
                None -> create patches
                a string -> previously generated patches folder name, e.g. 'new'
        :param label_type: 'str' or other
        :param size: width of a patch
        :param data_selection: must be in ['all', 'sup', 'usable']
                all -> all data
                sup -> data with supportive labels
                usable -> data with nonempty labels
        :return: label1, label2, cv_data
                label string is something like <Nucleus;Cytoplasm;Golgi apparatus> WITHOUT quote
                cv_data is in the shape of (500*270*size*size*3)
        """
        label1s = []
        label2s = []
        cv_data = []
        if size is not None:
            self.__set_width(int(int(size) / 2))
        dir_sets = os.listdir(self.__data_dir)
        dir_sets = [a_set for a_set in dir_sets if a_set.startswith('liver_ENSG')]

        xlsx_path = os.path.join(self.__base_dir, 'data_label.xlsx')
        book_sheet = xlrd.open_workbook(xlsx_path).sheet_by_index(0)
        sets = book_sheet.col_values(0)
        pos1 = book_sheet.col_values(6)
        pos2 = book_sheet.col_values(7)
        reli = book_sheet.col_values(8)

        # if my version of 'data_label.xlsx' or not
        if sets[0] == '基因编号':
            sets = sets[1:]
            pos1 = pos1[1:]
            pos2 = pos2[1:]
            reli = reli[1:]

        # remove <'> at head and tail, while "[]" becomes ""(empty string)
        for i in range(len(pos1)):
            # 这里有个很奇怪的bug，直接去头尾一个字符会导致有的字符串第一个字没了，所以写成这样
            pos1[i] = pos1[i].replace('\'', '').replace('[', '').replace(']', '')
            pos2[i] = pos2[i].replace('\'', '').replace('[', '').replace(']', '')
            sets[i] = sets[i].replace('\'', '').replace('[', '').replace(']', '')
            reli[i] = reli[i][1:-1]
        if data_selection == 'all':
            for a_set in dir_sets:
                ensg_index = a_set.split('_')[1]
                if exist is None:
                    a_full_set = self.__get_patch_a_set(a_set)
                else:
                    a_full_set = self.__get_generated_patch_a_set(a_set, exist)
                pos_index = sets.index(ensg_index)
                if label_type == 'str':
                    # yield pos1[pos_index].split(';'), pos2[pos_index].split(';'), a_full_set
                    label1s.append(pos1[pos_index].split(';'))
                    label2s.append(pos2[pos_index].split(';'))
                    cv_data.append(a_full_set)
                else:
                    label1 = [self.__labels[x] for x in pos1[pos_index].split(';')]
                    label2 = [self.__labels[x] for x in pos2[pos_index].split(';')]
                    # yield label1, label2, a_full_set
                    label1s.append(label1)
                    label2s.append(label2)
                    cv_data.append(a_full_set)
        elif data_selection == 'sup':
            suffices = book_sheet.col_values(2)[1:]
            for i in range(len(reli)):
                if reli[i] == 'Supportive':
                    if exist is None:
                        a_full_set = self.__get_patch_a_set('liver_{}_{}'.format(sets[i], suffices[i][1:-1]))
                    else:
                        a_full_set = self.__get_generated_patch_a_set('liver_{}_{}'.format(sets[i], suffices[i][1:-1]),
                                                                      exist)
                    if label_type == 'str':
                        # yield pos1[i].split(';'), pos2[i].split(';'), a_full_set
                        label1s.append(pos1[i].split(';'))
                        label2s.append(pos2[i].split(';'))
                        cv_data.append(a_full_set)
                    else:
                        label1 = [self.__labels[x] for x in pos1[i].split(';')]
                        label2 = [self.__labels[x] for x in pos2[i].split(';')]
                        # yield label1, label2, a_full_set
                        label1s.append(label1)
                        label2s.append(label2)
                        cv_data.append(a_full_set)
        elif data_selection == 'usable':
            suffices = book_sheet.col_values(2)[1:]
            for i in range(len(reli)):
                if reli[i] != '':
                    if exist is None:
                        a_full_set = self.__get_patch_a_set('liver_{}_{}'.format(sets[i], suffices[i][1:-1]))
                    else:
                        a_full_set = self.__get_generated_patch_a_set('liver_{}_{}'.format(sets[i], suffices[i][1:-1]),
                                                                      exist)
                    if label_type == 'str':
                        # yield pos1[i].split(';'), pos2[i].split(';'), a_full_set
                        label1s.append(pos1[i].split(';'))
                        label2s.append(pos2[i].split(';'))
                        cv_data.append(a_full_set)
                    else:
                        label1 = [self.__labels[x] for x in pos1[i].split(';')]
                        label2 = [self.__labels[x] for x in pos2[i].split(';')]
                        # yield label1, label2, a_full_set
                        label1s.append(label1)
                        label2s.append(label2)
                        cv_data.append(a_full_set)

        return label1s, label2s, cv_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image parser')
    parser.add_argument('-w', '--width',
                        dest='width',
                        help='width(for type square) or diameter(for type circle) of each slice')
    parser.add_argument('-f', '--folder-name',
                        dest='folder_name',
                        help='name of new directory, new folder will be in \'data/liver_XXX/<new_folder>\'')
    parser.add_argument('-b', '--base_dir',
                        dest='base_dir',
                        help='path of <HPA_ieee> directory. If this file is in ./HPA_ieee, then not required')
    args = parser.parse_args()

    # sample usage
    # image_pre = ImagePreprocessor(base_dir=args.base_dir)
    image_pre = ImagePreprocessor(base_dir='D:\\All_Projects\\ML_project\\HPA_ieee')
    # image_pre.generate_patches()
    # for l1, l2, d in image_pre.get_dataset_patched(size=20, data_selection='all', label_type='int', exist='new'):
    #     print(l1, l2, len(d), numpy.array(d).shape)
    count = 0
    for l1, l2, d in image_pre.get_dataset_full(data_selection='all', label_type='int'):
        print(len(l1), len(l2), numpy.array(d).shape, count)
        count += 1
