# coding: utf-8
from typing import Union
import numpy as np

from .ult import get_img_ndarray, show_img
from .preprocessing import fetch_single_hand_roi, rgb_to_hsv, grayscale, resize

"""
Types of preprocessing:

Base-Norm:
    Region Norm : ROI
    Illumination Norm: RGB -> HSV

Data-Aug:
    a. rotate
    b. flip
    c. noise
    d. filter
    e. dilation 
    f. erosion
    *. STL

Background-Norm:
    a. bg_normalization_red_channel()
    b. bg_normalization_fg_extraction()

Channel-Norm: grayscale

Resolution-Norm: resize
"""

"""
Pipeline Naming Convention:

Type-A: this type of pipeline doesn't contain any data augmentation structure
    pipeline_# : return np.ndarray

Type-B: this type of pipeline contains data augmentation structure
    pipeline_da_# : return [np.ndarray, ..., np.ndarray]

"""


def base_normalize(image: np.ndarray) -> Union[np.ndarray, None]:
    hand_roi = fetch_single_hand_roi(image)  # with default padding 15
    if hand_roi is None:
        msg = f"[WARN] - base normalized failed. By: failed to fetch hand-roi."
        print(msg)
        return

    # normalize illumination
    _, norm_hand, _ = rgb_to_hsv(hand_roi)  # with adaptive smooth

    return norm_hand


def channel_normalize(image: np.ndarray) -> np.ndarray:
    return grayscale(image)


def resolution_normalize(image: np.ndarray, size=28) -> np.ndarray:
    return resize(image, size)


# >>>>>>>>>>>> pipeline template >>>>>>>>>>>>
def pipeline_1(image: Union[np.ndarray, str]):
    """
    Base-Norm

    Channel-Norm

    Resolution-Norm

    :param image: np.ndarray | str
    :return:
    """
    image = get_img_ndarray(image)
    norm_hand = base_normalize(image)

    show_img(norm_hand)

# <<<<<<<<<<<< pipeline template <<<<<<<<<<<<
