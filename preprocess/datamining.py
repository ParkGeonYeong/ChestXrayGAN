from __future__ import print_function
from glob import glob
from PIL import Image

import tensorflow as tf
import shutil
from model.param import hps
import numpy as np
import os
import cv2


def gen_random(mode, size):
    # z = gen_random(self.param.z_mode, self.param.z_size)
    if mode == 'normal': return np.random.normal(0, 1, size=size).astype(np.float32)
    if mode == 'uniform_signed': return np.random.uniform(-1, 1, size=size).astype(np.float32)
    if mode == 'uniform_unsigned': return np.random.uniform(0, 1, size=size).astype(np.float32)


def crop_resize(filelist, crop_width, newpath):
    """
    crop_resize : crop image and take non-zero pixels

    :param img:
    tensor, (W, H, 1) for gray scale imgs
    :param ratio (from model.param):
    Ratio of survived non-zero pixels
    :param ths (from model.param):
    threshold of number of non-zero pixels to allow resize
    :param trashpath:
    path to save bad images

    :return:
    cropped and resized images. (W', H', 1)
    """
    print("INFO:resize started")
    for file in filelist:
        img = np.asarray(Image.open(file))
        img_number = file.split('filtered_img/')[1]
        nonzero_w = np.argwhere(img > 20)[:, 1]

        med_w = int(np.median(nonzero_w))
        if med_w < crop_width:
            med_w = int(img.shape[1] / 2)

        crop_l, crop_h = med_w-crop_width, med_w+crop_width
        img = img[crop_l:crop_h, crop_l:crop_h]
        img = cv2.resize(img, dsize=(hps.resize, hps.resize), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(newpath, img_number), img)
        print("INFO:{} resized".format(img_number))


def parse_png(img_file):
    image_string = tf.read_file(img_file)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    return image


def get_all_png(names, template):
    all_file = []
    for disease in names:
        all_file += glob(os.path.join(template, disease.replace('No Finding', 'No_Finding'), '*.png'))
    return all_file


def load_all(img_list):
    total_len = len(img_list)
    print("total length: ", total_len)
    print(img_list)
    img_list = tf.constant(img_list)

    dataset = tf.data.Dataset.from_tensor_slices(img_list)
    dataset = dataset.map(parse_png)
    dataset = dataset.shuffle(10000).repeat()
    dataset = dataset.batch(hps.batch_size)
    print('Dataset is built')

    return dataset, total_len


def make_path(pathlist):
    for path in pathlist:
        if not (os.path.exists(path)):
            os.mkdir(path)