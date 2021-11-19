# coding: utf-8
from typing import Union
import numpy as np

from .ult import get_img_ndarray, show_img
from .preprocessing import fetch_single_hand_roi, rgb_to_hsv


def base_normalize(image: np.ndarray) -> Union[np.ndarray, None]:
    hand_roi = fetch_single_hand_roi(image)  # with default padding 15
    if hand_roi is None:
        msg = f"[WARN] - base normalized failed. By: failed to fetch hand-roi."
        print(msg)
        return

    # normalize illumination
    _, norm_hand, _ = rgb_to_hsv(hand_roi)  # with adaptive smooth

    return norm_hand


def pipe_1(image: Union[np.ndarray, str]):
    image = get_img_ndarray(image)
    norm_hand = base_normalize(image)

    show_img(norm_hand)
