# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/21/21
"""
from imutils.paths import list_images
from tqdm import tqdm

from data_tool import *
from image_pipeline import get_img_ndarray


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


def drop_small_image():
    img_path_ls = list_images(DATASET2_DIR_PATH)

    for img_path in tqdm(img_path_ls):
        img = get_img_ndarray(img_path)

        if img is not None:
            resolution = img.shape[0] * img.shape[1]
            if resolution < 40000:
                print(f"remove: {img_path}")
                os.remove(img_path)


if __name__ == '__main__':
    DATASET2_DIR_PATH = 'dataset2'

    drop_small_image()
