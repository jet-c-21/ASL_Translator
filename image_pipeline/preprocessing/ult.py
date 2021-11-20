"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""
# coding: utf-8
import os
from typing import Union

import numpy as np
from cv2 import cv2
import PIL
from matplotlib import pyplot as plt


def get_img_ndarray(input_obj: Union[str, np.ndarray]) -> Union[np.ndarray, None]:
    if isinstance(input_obj, np.ndarray):
        return input_obj

    elif isinstance(input_obj, str):
        if os.path.exists(input_obj):
            try:
                return cv2.imread(input_obj)
            except Exception as e:
                msg = f"[WARN] - Failed to read image from path : {input_obj}. Error: {e}"
                print(msg)


def pil_image_to_np_ndarray(image: PIL.Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def np_ndarray_to_pil_image(image: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def show_img(image: np.ndarray, window_name=''):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def img_show(image: np.ndarray):
    plt.imshow(image)
    plt.show()


def img_plt_save(image: np.ndarray, fp='output.jpg'):
    plt.imshow(image)
    plt.savefig(fp)


def ls_to_chunks(ls: list, chunk_size=100) -> list:
    result = list()
    for i in range(0, len(ls), chunk_size):
        result.append(ls[i:i + chunk_size])

    return result
