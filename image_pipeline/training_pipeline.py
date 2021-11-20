# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""
from typing import Union
import numpy as np

from image_pipeline.preprocessing.ult import get_img_ndarray, show_img
from .preprocessing import fetch_single_hand_roi, rgb_to_hsv, grayscale, resize, bg_normalization_red_channel, \
    bg_normalization_fg_extraction, BgRemover, remove_bg, da_rotate, da_filter, da_add_noise, da_dilation, da_erosion
from .general_pipeline import roi_normalize, bg_normalize, illumination_normalize, channel_normalize, \
    resolution_normalize

from tqdm import tqdm


def t_pipeline_a(image: Union[np.ndarray, str], bgr: BgRemover, img_size=28):
    # load image
    image = get_img_ndarray(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to load image"
        print(msg)
        return

    # process image
    image = bg_normalize(image, bgr)

    image = roi_normalize(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to get norm_hand"
        print(msg)
        return

    image = illumination_normalize(image)
    image = channel_normalize(image)
    image = resolution_normalize(image, img_size)

    return image


def t_pipeline_with_da_1(image: Union[np.ndarray, str], bgr: BgRemover, img_size=28):
    img_original_ls = list()
    img_output_ls = list()

    image = get_img_ndarray(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to load image"
        print(msg)
        return

    img_original_ls.append(image)

    # rotate positive range(5, 35, 5)
    for deg in range(5, 35, 5):
        img_original_ls.append(da_rotate(image, deg))

    # rotate negative range(5, 35, 5)
    for deg in range(-5, -35, -5):
        img_original_ls.append(da_rotate(image, deg))

    # noise - speckle
    img_original_ls.append(da_add_noise(image, 'speckle'))

    # noise - gaussian
    img_original_ls.append(da_add_noise(image, 'gaussian'))

    # noise - poisson
    img_original_ls.append(da_add_noise(image, 'poisson'))

    # filter - blur
    img_original_ls.append(da_filter(image, 'blur'))

    # dilation
    img_original_ls.append(da_dilation(image))

    # erosion
    img_original_ls.append(da_dilation(image))

    for img in img_original_ls:
        aug_img = t_pipeline_a(img, bgr)
        if aug_img is not None:
            show_img(aug_img)
        # img_output_ls.append()

    # for a, b in zip(img_original_ls, img_output_ls):
    #     show_img(a)
    #     show_img(b)


def t_pipeline_with_da_2(image: Union[np.ndarray, str], bgr: BgRemover, img_size=28) -> Union[list, None]:
    img_base_with_deg = list()
    img_output_ls = list()

    image = get_img_ndarray(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to load image"
        print(msg)
        return

    base_image = roi_normalize(image)
    if base_image is None:
        return

    base_image = illumination_normalize(image)

    # rotate positive range
    for deg in range(1, 20, 1):
        img_base_with_deg.append(da_rotate(base_image, deg))

    # rotate negative range
    for deg in range(-1, -20, -1):
        img_base_with_deg.append(da_rotate(base_image, deg))

    aug_img_ls = list()
    for img in img_base_with_deg:
        # noise - speckle
        aug_img_ls.append(da_add_noise(img, 'speckle'))

        # noise - gaussian
        aug_img_ls.append(da_add_noise(img, 'gaussian'))

        # noise - poisson
        aug_img_ls.append(da_add_noise(img, 'poisson'))

        # filter - blur
        aug_img_ls.append(da_filter(img, 'blur'))

        # dilation
        aug_img_ls.append(da_dilation(img))

        # erosion
        aug_img_ls.append(da_dilation(img))

    for img in aug_img_ls:
        f_aug_img = bg_normalize(img, bgr)
        f_aug_img = channel_normalize(f_aug_img)
        f_aug_img = resolution_normalize(f_aug_img)
        img_output_ls.append(f_aug_img)

    return img_output_ls
