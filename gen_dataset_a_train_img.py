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

from data_tool import create_dir, get_img_save_name
from image_pipeline import *
from image_pipeline.preprocessing import ls_to_chunks


# @profile
def create_norm_hand(image_path: str, alphabet: str, bgr: BgRemover) -> tuple:
    output_img_dict = dict()
    raw_image = get_img_ndarray(image_path)
    if raw_image is None:
        return False, output_img_dict

    norm_hand = t_pipeline_a(raw_image, hdt, bgr)
    if norm_hand is None:
        return False, output_img_dict

    raw_image_s_dir = f"{OUTPUT_DIR_ROOT}/{alphabet}"
    img_name = get_img_save_name(alphabet)
    raw_image_s_path = f"{raw_image_s_dir}/{img_name}.jpg"
    output_img_dict['raw'] = (raw_image_s_path, raw_image)

    norm_hand_s_dir = f"{OUTPUT_AP_DIR_ROOT}/{alphabet}"
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

    img_raw_s_dir = f"{OUTPUT_DIR_ROOT}/{alphabet}"
    create_dir(img_raw_s_dir)
    norm_hand_s_dir = f"{OUTPUT_AP_DIR_ROOT}/{alphabet}"
    create_dir(norm_hand_s_dir)

    alphabet_img_chunks = ls_to_chunks(alphabet_img_ls, CHUNK_SIZE)

    for i, img_chunk in enumerate(alphabet_img_chunks, start=1):
        task_res = img_chunk_processor(alphabet, img_chunk)
        task_res = record_failed_task_and_del(alphabet, task_res)

        save_task_result_mp(task_res)
        print(f"Fin {alphabet} - chunks: ({i}/{len(alphabet_img_chunks)}) \n")
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


def main():
    create_save_dir(OUTPUT_DIR_ROOT)
    create_save_dir(OUTPUT_AP_DIR_ROOT)
    alphabet_dir_ls = get_alphabet_dir_ls()
    tidy_up(alphabet_dir_ls)


if __name__ == '__main__':
    DATASET1_DIR_PATH = 'dataset1'

    OUTPUT_DIR_ROOT = 'TRAIN_DATASET_A'

    OUTPUT_AP_DIR_ROOT = 'TRAIN_DATASET_A_AP'

    CHUNK_SIZE = 300

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
