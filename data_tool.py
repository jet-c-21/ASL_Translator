# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/11/21
"""
import os
from image_pipeline.preprocessing import gen_random_token


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_img_save_name(alphabet, is_train=True, rt_len=12):
    if is_train:
        return f"TRAIN_{alphabet}_{gen_random_token(rt_len)}"
    else:
        return f"TEST_{alphabet}_{gen_random_token(rt_len)}"
