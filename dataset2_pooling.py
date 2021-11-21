# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/21/21
"""
from imutils.paths import list_images
from data_tool import *


def rename_sub_folder():
    for sub_dir in os.listdir(DATASET2_DIR_PATH):
        sub_dir_old_name = sub_dir
        sub_dir_new_name = sub_dir_old_name.upper()

        sub_dir_old_path = f"{DATASET2_DIR_PATH}/{sub_dir_old_name}"
        sub_dir_new_path = f"{DATASET2_DIR_PATH}/{sub_dir_new_name}"

        os.rename(sub_dir_old_path, sub_dir_new_path)


def remove_duplicated_img():
    img_path_ls = list(list_images(DATASET2_DIR_PATH))
    for img_path in img_path_ls:
        img_name = img_path.split('/')[-1]
        if 'copy' in img_name:
            os.remove(img_path)


def remove_cropped_img():
    img_path_ls = list(list_images(DATASET2_DIR_PATH))
    for img_path in img_path_ls:
        img_name = img_path.split('/')[-1]
        if 'cropped' in img_name:
            os.remove(img_path)


if __name__ == '__main__':
    OUTPUT_DIR_ROOT = 'TEST_DATASET_A'
    OUTPUT_AP_DIR_ROOT = 'TEST_DATASET_A_AP'

    DATASET2_DIR_PATH = 'dataset2'

    x = list(list_images(DATASET2_DIR_PATH))
    print(len(x))
