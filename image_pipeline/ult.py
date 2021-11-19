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


def show_img(image: np.ndarray):
    cv2.imshow('', image)
    cv2.waitKey(0)
