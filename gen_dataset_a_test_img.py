# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/20/21
"""
from collections import Counter

# from tqdm import tqdm
import datetime
from cv2 import cv2
from imutils.paths import list_images

from data_tool import *
from image_pipeline import *

dataset4_p_failed = dict()


def create_norm_hand(image_path: str, alphabet: str, hdt: HandDetector, bgr: BgRemover) -> bool:
    raw_image = get_img_ndarray(image_path)
    if raw_image is None:
        return False

    norm_hand = pipeline_base(raw_image, hdt, bgr)
    if norm_hand is None:
        return False

    raw_image_s_dir = f"{OUTPUT_DIR_ROOT}/{alphabet}"
    img_name = get_img_save_name(alphabet)
    raw_image_s_path = f"{raw_image_s_dir}/{img_name}.jpg"
    cv2.imwrite(raw_image_s_path, raw_image)

    norm_hand_s_dir = f"{OUTPUT_AP_DIR_ROOT}/{alphabet}"
    norm_hand_s_path = f"{norm_hand_s_dir}/{img_name}.jpg"
    cv2.imwrite(norm_hand_s_path, norm_hand)

    return True


def record_pipeline_res(alphabet: str, task_res_dict: dict, dataset_dict: dict):
    if task_res_dict.get(False):
        f_count = task_res_dict.get(False)
    else:
        f_count = 0

    if alphabet in dataset_dict.keys():
        dataset_dict[alphabet] += f_count
    else:
        dataset_dict[alphabet] = f_count


def handle_alphabet(alphabet_img_ls: list):
    alphabet = alphabet_img_ls[0].split('/')[-2]

    img_raw_s_dir = f"{OUTPUT_DIR_ROOT}/{alphabet}"
    create_dir(img_raw_s_dir)

    norm_hand_s_dir = f"{OUTPUT_AP_DIR_ROOT}/{alphabet}"
    create_dir(norm_hand_s_dir)

    task_result = list()
    for img_path in tqdm(alphabet_img_ls, total=len(alphabet_img_ls)):
        res = create_norm_hand(img_path, alphabet, hdt, bgr)
        task_result.append(res)
        # break

    task_res_dict = dict(Counter(task_result))
    record_pipeline_res(alphabet, task_res_dict, dataset4_p_failed)

    print(f"Fin {alphabet} \n")


def tidy_up(alphabet_dir_ls: list):
    for i, alphabet_dir in enumerate(alphabet_dir_ls, start=1):
        alphabet_img_ls = list(list_images(alphabet_dir))
        alphabet_img_ls.sort()
        handle_alphabet(alphabet_img_ls)
        # break


def process_dataset4():
    s = datetime.datetime.now()
    alphabet_dir_ls = get_alphabet_dir_ls(DATASET4_DIR_PATH)
    tidy_up(alphabet_dir_ls)
    e = datetime.datetime.now()
    exe_time = e - s

    csv_prefix = 'DATASET_A_TEST_dataset4_'
    create_p_res_csv(exe_time, dataset4_p_failed, prefix=csv_prefix)





if __name__ == '__main__':
    OUTPUT_DIR_ROOT = 'TEST_DATASET_A'
    OUTPUT_AP_DIR_ROOT = 'TEST_DATASET_A_AP'
    create_dir(OUTPUT_DIR_ROOT)
    create_dir(OUTPUT_AP_DIR_ROOT)

    hdt = HandDetector()

    bgr = BgRemover()
    bgr.load_model()

    DATASET2_DIR_PATH = 'dataset2'
    # rename_sub_folder()
    # remove_duplicated_img()
    # remove_cropped_img()

    DATASET4_DIR_PATH = 'dataset4'
    process_dataset4()
