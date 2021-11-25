# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/20/21
"""
import multiprocessing as mp
import os
import datetime

import pandas as pd
from cv2 import cv2
from imutils.paths import list_images

from image_pipeline import *
from image_pipeline.preprocessing import ls_to_chunks, gen_random_token

from memory_profiler import profile
from data_tool import create_dir, get_img_save_name

import gc
from functools import partial

from tqdm import tqdm


def _task_worker(image_path: str, alphabet: str, hdt: HandDetector, bgr: BgRemover) -> bool:
    raw_image = get_img_ndarray(image_path)
    if raw_image is None:
        return False

    norm_hand = t_pipeline_a(raw_image, hdt, bgr)
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


def images_processor_mp(alphabet: str, img_path_ls: list) -> list:
    with mp.Pool() as pool:
        task = partial(_task_worker, alphabet=alphabet, hdt=hdt, bgr=bgr)
        task_result = list(
            tqdm(
                pool.imap(task, img_path_ls), total=len(img_path_ls)
            )
        )

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

    img_raw_s_dir = f"{OUTPUT_DIR_ROOT}/{alphabet}"
    create_dir(img_raw_s_dir)
    norm_hand_s_dir = f"{OUTPUT_AP_DIR_ROOT}/{alphabet}"
    create_dir(norm_hand_s_dir)

    task_res = images_processor_mp(alphabet, alphabet_img_ls)

    print(task_res)

    # alphabet_img_chunks = ls_to_chunks(alphabet_img_ls, CHUNK_SIZE)
    #
    # for i, img_chunk in enumerate(alphabet_img_chunks, start=1):
    #     task_res = img_chunk_processor(alphabet, img_chunk)
    #     # task_res = record_failed_task_and_del(alphabet, task_res)
    #
    #     # save_task_result_mp(task_res)
    #     # print(f"Fin {alphabet} - chunks: ({i}/{len(alphabet_img_chunks)}) \n")
    #
    #     break


def tidy_up(alphabet_dir_ls: list):
    for alphabet_dir in alphabet_dir_ls:
        alphabet_img_ls = list(list_images(alphabet_dir))
        alphabet_img_ls.sort()

        handle_alphabet(alphabet_img_ls)
        break


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


def main():
    create_save_dir(OUTPUT_DIR_ROOT)
    create_save_dir(OUTPUT_AP_DIR_ROOT)
    alphabet_dir_ls = get_alphabet_dir_ls()
    tidy_up(alphabet_dir_ls)


if __name__ == '__main__':
    DATASET1_DIR_PATH = '../dataset1'

    OUTPUT_DIR_ROOT = '../TRAIN_DATASET_A'

    OUTPUT_AP_DIR_ROOT = '../TRAIN_DATASET_A_AP'

    CHUNK_SIZE = 2

    pipeline_failed = dict()

    hdt = HandDetector()

    bgr = BgRemover()
    bgr.load_model()

    s = datetime.datetime.now()
    main()
    e = datetime.datetime.now()
    exe_time = e - s

    seconds = exe_time.total_seconds()
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    exe_time_str = f"{hours}hr_{minutes}m_{seconds}s"
    print(f"time cost = {exe_time_str}")

    csv_s_path = f"DATASET_A_train_img_failed_cost={exe_time_str}.cvs"

    df = pd.DataFrame(list(pipeline_failed.items()), columns=['Alphabet', 'FailedCount'])
    df.to_csv(csv_s_path, index=False)
