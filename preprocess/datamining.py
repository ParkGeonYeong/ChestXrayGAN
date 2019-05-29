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


def crop_resize(filelist, trashpath):
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
    if not os.path.exists(trashpath):
        os.mkdir(trashpath)

    trash = []
    for file in filelist:
        img = np.asarray(Image.open(file))

        nonzero_w = np.argwhere(img > 30)[1]
        max_entity = np.max(nonzero_w)-np.min(nonzero_w)
        med_w = np.median(nonzero_w)

        if max_entity < hps.resize_ths:
            shutil.move(file, trashpath)
            trash.append(file)
            continue

        crop_l, crop_h = med_w-400, med_w+400
        img = img[crop_l:crop_h, crop_l:crop_h]
        img = cv2.resize(img, dsize=(hps.resize, hps.resize), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(file, img)

    return list(set(filelist)-set(trash))


def parse_png(img_file):
    image_string = tf.read_file(img_file)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    image = crop_resize(image)
    return image


def get_all_png(names, template):
    all_file = []
    for disease in names:
        all_file += glob(os.path.join(template, disease))
    return all_file


def load_all(img_list):
    total_len = len(img_list)
    print("total length: ", total_len)

    img_list = tf.constant(img_list)

    dataset = tf.data.Dataset.from_tensor_slices(img_list)
    dataset = dataset.map(parse_png)
    dataset = dataset.repeat()
    dataset = dataset.batch(hps.batch)
    print('Dataset is built')

    return dataset, total_len


