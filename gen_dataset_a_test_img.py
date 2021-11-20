# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/20/21
"""
from imutils.paths import list_images
from image_pipeline import *
from image_pipeline.preprocessing import ls_to_chunks, gen_random_token
from string import ascii_uppercase
import os
import multiprocessing as mp
from cv2 import cv2
# from tqdm import tqdm
from memory_tool import memory
from functools import partial
from image_pipeline.preprocessing import has_single_hand
from collections import Counter
import sys
from pprint import pp
import pandas as pd
from imutils.paths import list_images

pipeline_failed = dict()


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_img_save_name(alphabet, is_train=True, rt_len=12):
    if is_train:
        return f"TRAIN_{alphabet}_{gen_random_token(rt_len)}"
    else:
        return f"TEST_{alphabet}_{gen_random_token(rt_len)}"


def main():
    # create_save_dir(OUTPUT_DIR_ROOT)
    # create_save_dir(OUTPUT_AP_DIR_ROOT)
    # alphabet_dir_ls = get_alphabet_dir_ls()
    # tidy_up(alphabet_dir_ls)
    pass


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
    DATASET2_DIR_PATH = 'dataset2'

    OUTPUT_DIR_ROOT = 'DATASET_A'

    OUTPUT_AP_DIR_ROOT = 'DATASET_A_AP'

    # rename_sub_folder()
    # remove_duplicated_img()
    remove_cropped_img()

    # bgr = BgRemover()
    # bgr.load_model()
    # main()

    # df = pd.DataFrame(
    #     list(pipeline_failed.items()),
    #     columns=['Alphabet', 'FailedCount']
    # )
    # df.to_csv('DATASET_A_train_img_failed.cvs', index=False)
