from __future__ import print_function
import pandas as pd
import numpy as np
import os
from shutil import copyfile
import argparse
import tensorflow
from model.param import hps


disease_name = [
    'Hernia',
    'Pneumonia',
    'Fibrosis',
    'Edema',
    'Emphysema',
    'Cardiomegaly',
    'Pleural_Thickening',
    'Consolidation',
    'Pneumothorax',
    'Mass',
    'Nodule',
    'Atelectasis',
    'Effusion',
    'Infiltration',
    'No Finding',
]

def read_csv_and_make_dir(csv, save_path_template=None, image_path=None, selected_disease=None):
    """
    read_csv_and_make_dir: read sample_labels csv file and save each corresponding images to label folder

    :param csv:
    str, your directory of 'sample_labels.csv' file

    :param save_path_template:
    str, your template of directory to save images. Name of disease will be concatenated later. You do not have to input existing directory only
    ex) /home/usr/training

    :param image_path:
    str, your directory of whole image dataset.
    ex) /home/usr/data/images

    :param selected_disease:
    list, selected name of disease for training
    ex) ['Atelectasis', 'Consolidation', 'Emphysema']

    :return:
    Saved folders
    ex) /home/usr/training/Atelectasis/many_images..
        /home/usr/training/Consolidation/many_images..
        ...
    """
    if not (os.path.exists(save_path_template)):
        os.mkdir(save_path_template)

    original_file = pd.read_csv(csv)
    assert len(original_file.index) == 5606
    make_path(save_path_template, selected_disease)

    disease_label_column = list(original_file.Finding_Labels)
    image_index_column = list(original_file.Image_Index)

    print('INFO:start reading images...')
    for ind, disease in enumerate(disease_label_column):
        many_disease_or_not = disease.split('|')
        for word in many_disease_or_not:
            if not word in selected_disease:
                continue
            path_to_save = os.path.join(save_path_template, word.replace('No Finding', 'No_Finding'), image_index_column[ind])
            img_tobe_load = os.path.join(image_path, image_index_column[ind])
            copyfile(img_tobe_load, path_to_save)
    print('INFO:done')


def make_path(path_template, selected_disease):
    """
    make_path: make folders in order of path_template+selected_disease

    :param path_template:
    str, your template of directory
    ex) /home/usr/training

    :param selected_disease:
    list, selected name of disease for training
    ex) ['Atelectasis', 'Consolidation', 'Emphysema']

    :return:
    ex) /home/usr/training/Atelectasis, /home/usr/training/Consolidation, /home/usr/training/Emphysema
    """
    for disease in selected_disease:
        if disease == 'No Finding':
            disease = 'No_Finding'
            # mkdir do not allow space within word
        new_path = os.path.join(path_template, disease)
        if not (os.path.exists(new_path)):
            os.mkdir(new_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='require file path')
    parser.add_argument('--save_path_template', required=True, help='질병별로 폴더를 생성할 디렉토리, 예시 /home/usr/training')
    parser.add_argument('--image_path', required=True, help='이미지 데이터셋 저장한 곳, 예시 /home/usr/images')
    parser.add_argument('--csv_path', required=True, help='csv 저장한 곳, 예시 /home/usr/data/example.csv')
    args = parser.parse_args()

    # Change selected_disease what you want
    read_csv_and_make_dir(args.csv_path, save_path_template=args.save_path_template, image_path=args.image_path,
                          selected_disease=hps.disease_group)
