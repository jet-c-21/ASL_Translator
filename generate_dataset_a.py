# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/20/21
"""
from imutils.paths import list_images
from image_pipeline import *
from string import ascii_uppercase
import os
from cv2 import cv2
# from tqdm import tqdm
from memory_tool import memory
import sys
image_idx = 0


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def handle_alphabet(alphabet_dir: list):
    global image_idx
    alphabet = alphabet_dir[0].split('/')[-2]

    img_raw_s_dir = f"{OUTPUT_DIR_ROOT}/train/{alphabet}"
    create_dir(img_raw_s_dir)
    norm_hand_s_dir = f"{OUTPUT_AP_DIR_ROOT}/train/{alphabet}"
    create_dir(norm_hand_s_dir)

    # for image_path in tqdm(alphabet_dir, desc=f"[{alphabet}]", total=len(alphabet_dir)):
    for i, image_path in enumerate(alphabet_dir):
        img_raw = get_img_ndarray(image_path)
        norm_hand = pipeline_base(img_raw, bgr)
        if norm_hand is not None:
            image_idx += 1
            img_name = f"TRAIN_{alphabet}_{image_idx}.jpg"

            # same raw img to DATASET_*
            img_raw_s_path = f"{img_raw_s_dir}/{img_name}"
            cv2.imwrite(img_raw_s_path, img_raw)

            # same piped img to DATASET_*_AP
            norm_hand_s_path = f"{norm_hand_s_dir}/{img_name}"
            cv2.imwrite(norm_hand_s_path, norm_hand)

        print(f"fin : ({i}/{len(alphabet_dir)})")


def tidy_up(alphabet_dir_ls: list):
    for alphabet_dir in alphabet_dir_ls:
        alphabet_dir = list(list_images(alphabet_dir))
        alphabet_dir.sort()

        handle_alphabet(alphabet_dir)
        break


def get_alphabet_dir_ls():
    result = list()
    for alphabet_dir_name in os.listdir(DATASET1_DIR_PATH):
        if alphabet_dir_name in ['del', 'nothing', 'space']:
            continue
        alphabet_dir_path = f"{DATASET1_DIR_PATH}/{alphabet_dir_name}"
        result.append(alphabet_dir_path)

    return sorted(result)


def main():
    alphabet_dir_ls = get_alphabet_dir_ls()
    tidy_up(alphabet_dir_ls)


if __name__ == '__main__':
    DATASET1_DIR_PATH = 'dataset1/asl_alphabet_train'
    OUTPUT_DIR_ROOT = 'DATASET_A'
    OUTPUT_AP_DIR_ROOT = 'DATASET_A_AP'

    bgr = BgRemover()
    bgr.load_model()

    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
