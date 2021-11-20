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

pipeline_failed = dict()


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_img_save_name(alphabet, is_train=True, rt_len=12):
    if is_train:
        return f"TRAIN_{alphabet}_{gen_random_token(rt_len)}"
    else:
        return f"TEST_{alphabet}_{gen_random_token(rt_len)}"


def create_norm_hand(image_path: str, alphabet: str, bgr: BgRemover) -> tuple:
    output_img_dict = dict()
    raw_image = get_img_ndarray(image_path)
    if raw_image is None:
        return False, output_img_dict

    norm_hand = t_pipeline_a(raw_image, bgr)
    if norm_hand is None:
        return False, output_img_dict

    raw_image_s_dir = f"{OUTPUT_DIR_ROOT}/train/{alphabet}"
    img_name = get_img_save_name(alphabet)
    raw_image_s_path = f"{raw_image_s_dir}/{img_name}.jpg"
    output_img_dict['raw'] = (raw_image_s_path, raw_image)

    norm_hand_s_dir = f"{OUTPUT_AP_DIR_ROOT}/train/{alphabet}"
    norm_hand_s_path = f"{norm_hand_s_dir}/{img_name}.jpg"
    output_img_dict['norm'] = (norm_hand_s_path, norm_hand)

    return True, output_img_dict


def _task_result_helper(record: dict):
    cv2.imwrite(record['raw'][0], record['raw'][1])
    cv2.imwrite(record['norm'][0], record['norm'][1])


def save_task_result_mp(task_result: list):
    with mp.Pool() as pool:
        _ = list(
            tqdm(
                pool.imap(_task_result_helper, task_result), total=len(task_result)
            )
        )


def img_chunk_processor(alphabet: str, img_chunk: list) -> list:
    task_result = list()
    for img_path in tqdm(img_chunk, total=len(img_chunk)):
        task_result.append(create_norm_hand(img_path, alphabet, bgr))
    return task_result


def record_failed_task_and_del(alphabet: str, task_result: list) -> list:
    global pipeline_failed
    f_count = 0
    result = list()
    for r in task_result:
        if r[0]:
            result.append(r[1])
        else:
            f_count += 1

    if alphabet in pipeline_failed.keys():
        pipeline_failed[alphabet] += f_count
    else:
        pipeline_failed[alphabet] = f_count

    return result


def handle_alphabet(alphabet_img_ls: list):
    alphabet = alphabet_img_ls[0].split('/')[-2]

    img_raw_s_dir = f"{OUTPUT_DIR_ROOT}/train/{alphabet}"
    create_dir(img_raw_s_dir)
    norm_hand_s_dir = f"{OUTPUT_AP_DIR_ROOT}/train/{alphabet}"
    create_dir(norm_hand_s_dir)

    alphabet_img_chunks = ls_to_chunks(alphabet_img_ls, 300)

    for i, img_chunk in enumerate(alphabet_img_chunks, start=1):
        task_res = img_chunk_processor(alphabet, img_chunk)
        task_res = record_failed_task_and_del(alphabet, task_res)

        save_task_result_mp(task_res)
        print(f"Fin {alphabet}-chunks: ({i}/{len(alphabet_img_chunks)})")
        # break


def tidy_up(alphabet_dir_ls: list):
    for alphabet_dir in alphabet_dir_ls:
        alphabet_img_ls = list(list_images(alphabet_dir))
        alphabet_img_ls.sort()

        handle_alphabet(alphabet_img_ls)
        # break


def get_alphabet_dir_ls():
    result = list()
    for alphabet_dir_name in os.listdir(DATASET1_DIR_PATH):
        if alphabet_dir_name in ['del', 'nothing', 'space']:
            continue
        alphabet_dir_path = f"{DATASET1_DIR_PATH}/{alphabet_dir_name}"
        result.append(alphabet_dir_path)

    return sorted(result)


def create_save_dir(out_put_dir: str):
    if not os.path.exists(out_put_dir):
        create_dir(out_put_dir)
        create_dir(f"{out_put_dir}/train")
        create_dir(f"{out_put_dir}/test")


def main():
    create_save_dir(OUTPUT_DIR_ROOT)
    create_save_dir(OUTPUT_AP_DIR_ROOT)
    alphabet_dir_ls = get_alphabet_dir_ls()
    tidy_up(alphabet_dir_ls)


if __name__ == '__main__':
    DATASET1_DIR_PATH = 'dataset1/asl_alphabet_train'

    OUTPUT_DIR_ROOT = 'DATASET_A'

    OUTPUT_AP_DIR_ROOT = 'DATASET_A_AP'

    bgr = BgRemover()
    bgr.load_model()
    main()

    df = pd.DataFrame(
        list(pipeline_failed.items()),
        columns=['Alphabet', 'FailedCount']
    )
    df.to_csv('DATASET_A_train_img_failed.cvs', index=False)
