# coding: utf-8
from typing import Union
import numpy as np

from image_pipeline.preprocessing.ult import get_img_ndarray, show_img
from .preprocessing import fetch_single_hand_roi, rgb_to_hsv, grayscale, resize, bg_normalization_red_channel, \
    bg_normalization_fg_extraction, HandDetector, BgRemover, remove_bg, has_single_hand

"""
Types of preprocessing:

ROI-Norm : Hand-ROI

Illumination Norm: RGB-to-HSV (adaptive smooth)

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


def roi_normalize(image: np.ndarray, hdt: HandDetector) -> Union[np.ndarray, None]:
    hand_roi = fetch_single_hand_roi(image, hdt)  # with default padding 15
    if hand_roi is None:
        msg = f"[WARN] - roi-normalize failed. By: failed to fetch hand-roi."
        print(msg)
        return

    return hand_roi


def illumination_normalize(image: np.ndarray) -> np.ndarray:
    _, hand_roi, _ = rgb_to_hsv(image)  # with adaptive smooth

    return hand_roi


def bg_normalize_no_model(image: np.ndarray, mode='fg') -> np.ndarray:
    if mode == 'fg':
        return bg_normalization_fg_extraction(image)
    else:
        return bg_normalization_red_channel(image)


def bg_normalize(image: np.ndarray, bgr: BgRemover) -> np.ndarray:
    return remove_bg(image, bgr)


def channel_normalize(image: np.ndarray) -> np.ndarray:
    return grayscale(image)


def resolution_normalize(image: np.ndarray, size=28) -> np.ndarray:
    return resize(image, size)


# >>>>>>>>>>>> general pipeline template >>>>>>>>>>>>
def pipeline_0(image: Union[np.ndarray, str], hdt: HandDetector, img_size=28):
    image = get_img_ndarray(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to load image"
        print(msg)
        return

    image = roi_normalize(image, hdt)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to get norm_hand"
        print(msg)
        return

    image = illumination_normalize(image)
    image = channel_normalize(image)
    image = resolution_normalize(image, img_size)

    return image


def pipeline_2(image: Union[np.ndarray, str], bgr: BgRemover, img_size=28):
    # load image
    image = get_img_ndarray(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to load image"
        print(msg)
        return

    # process image
    # image = roi_normalize(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to get norm_hand"
        print(msg)
        return

    image = illumination_normalize(image)
    image = bg_normalize(image, bgr)
    image = channel_normalize(image)
    image = resolution_normalize(image, img_size)

    return image


def pipeline_5(image: Union[np.ndarray, str], hdt: HandDetector, bgr: BgRemover, img_size=28):
    # load image
    image = get_img_ndarray(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to load image"
        print(msg)
        return

    # process image
    image = roi_normalize(image, hdt)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to get norm_hand"
        print(msg)
        return

    image = bg_normalize(image, bgr)
    image = illumination_normalize(image)
    image = channel_normalize(image)
    image = resolution_normalize(image, img_size)

    return image


def pipeline_base(image: Union[np.ndarray, str], hdt: HandDetector,
                  bgr: BgRemover, img_size=28) -> Union[np.ndarray, None]:
    """
    ROI-Norm

    Background-Norm

    *Hand-Detection-Filter

    Illumination-Norm

    Channel-Norm

    Resolution-Norm

    :param image:
    :param hdt:
    :param bgr:
    :param img_size:
    :return: Union[np.ndarray, None]
    """
    # load image
    image = get_img_ndarray(image)
    if image is None:
        msg = f"[PIPE-WARN] - (1) - failed to pass pipeline_base. By: failed to load image"
        print(msg)
        return

    # process image
    image = roi_normalize(image, hdt)
    if image is None:
        msg = f"[PIPE-WARN] - (2) - failed to pass pipeline_base. By: failed to get norm_hand"
        print(msg)
        return

    image = bg_normalize(image, bgr)

    if not has_single_hand(image, hdt):
        msg = f"[PIPE-WARN] - (3) - failed to pass pipeline_base. By: can't detect any hand in the image, after bg_norm"
        print(msg)
        return

    image = illumination_normalize(image)
    image = channel_normalize(image)
    image = resolution_normalize(image, img_size)

    return image


def pipeline_app(image: Union[np.ndarray, str], hdt: HandDetector,
                 bgr: BgRemover, img_size=28) -> Union[np.ndarray, None]:
    """
    ROI-Norm

    Background-Norm

    *Hand-Detection-Filter

    Illumination-Norm

    Channel-Norm

    Resolution-Norm

    :param image:
    :param hdt:
    :param bgr:
    :param img_size:
    :return: Union[np.ndarray, None]
    """
    # load image
    image = get_img_ndarray(image)
    if image is None:
        msg = f"[PIPE-WARN] - (1) - failed to pass pipeline_base. By: failed to load image"
        print(msg)
        return

    # process image
    image = roi_normalize(image, hdt)
    if image is None:
        msg = f"[PIPE-WARN] - (2) - failed to pass pipeline_base. By: failed to get norm_hand"
        print(msg)
        return

    image = bg_normalize(image, bgr)

    image = illumination_normalize(image)
    image = channel_normalize(image)
    image = resolution_normalize(image, img_size)

    return image

# <<<<<<<<<<<< general pipeline template <<<<<<<<<<<<
