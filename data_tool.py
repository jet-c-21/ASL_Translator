# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/11/21
"""
import os
from image_pipeline.preprocessing import gen_random_token
from datetime import timedelta
import pandas as pd


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_img_save_name(alphabet, is_train=True, rt_len=12):
    if is_train:
        return f"TRAIN_{alphabet}_{gen_random_token(rt_len)}"
    else:
        return f"TEST_{alphabet}_{gen_random_token(rt_len)}"


def get_alphabet_dir_ls(root_dir):
    result = list()
    for alphabet_dir_name in os.listdir(root_dir):
        alphabet_dir_path = f"{root_dir}/{alphabet_dir_name}"
        result.append(alphabet_dir_path)
    return sorted(result)


def create_p_res_csv(exe_time: timedelta, dataset_p_failed_dict: dict, prefix=''):
    seconds = exe_time.total_seconds()
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    exe_time_str = f"{hours}hr_{minutes}m_{seconds}s"
    print(f"time cost = {exe_time_str}")

    df = pd.DataFrame(list(dataset_p_failed_dict.items()), columns=['Alphabet', 'FailedCount'])

    csv_s_path = f"{prefix}_img_pipe_f--cost={exe_time_str}.cvs"
    print(f"create {csv_s_path}")
    df.to_csv(csv_s_path, index=False)
