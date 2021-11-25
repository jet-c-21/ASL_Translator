# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/11/23
"""
from cv2 import cv2
import numpy as np
from typing import Union
import os
import zipfile
import gdown
from tqdm import tqdm

COLOR_MAP = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'pink': (255, 223, 220),
    'tiff_blue': (208, 216, 129)
}


def get_color(color_obj: Union[str, tuple]) -> tuple:
    if isinstance(color_obj, str):
        return COLOR_MAP[color_obj]
    elif isinstance(color_obj, tuple):
        return color_obj
    else:
        return COLOR_MAP['red']


def add_text_in_frame(img: np.ndarray, text, color='green', coord=(75, 75)) -> np.ndarray:
    text = str(text)
    color = get_color(color)
    size = 2
    thick = 2
    line_style = cv2.LINE_AA
    font_style = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, coord,
                font_style, size,
                color, thick,
                line_style)
    return img


def extract_zip(zip_path: str, extract_path=None):
    print('Extracting zip...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            if extract_path:
                zip_ref.extract(member=file, path=extract_path)
            else:
                zip_ref.extract(member=file, path='.')
    print('Finish extracting')


def download_gd_zip(file_id: str) -> str:
    d_url = f"https://drive.google.com/uc?id={file_id}"
    zip_path = f"{file_id}.zip"
    gdown.download(d_url, zip_path, quiet=False)
    return zip_path


def download_file_by_gdown(file_id: str, save_fp=None):
    zip_path = download_gd_zip(file_id)
    extract_zip(zip_path, save_fp)
    os.remove(zip_path)


def download_model_by_gdown():
    model_file_id = '1lWWL7HXz6V5OGDSk7bpkcQecEnUis12y'
    download_file_by_gdown(model_file_id)


def check_model_exist(model_dir='./ASLT-Model'):
    model_path = f"{model_dir}/app-model/saved_model.pb"
    if not os.path.isfile(model_path):
        # print(os.listdir(f"{model_dir}/app-model"))
        print('Downloading ASLT-Model...')
        download_model_by_gdown()
    else:
        print(f"[INFO] - ASLT-Model Dir Founded")
