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
    bg_normalization_fg_extraction, HandDetector, BgRemover, remove_bg, da_rotate, da_filter, da_add_noise, da_dilation, \
    da_erosion, has_single_hand
from .general_pipeline import roi_normalize, bg_normalize, illumination_normalize, channel_normalize, \
    resolution_normalize

from tqdm import tqdm

from cv2 import cv2


def t_pipeline_a(image: Union[np.ndarray, str],
                 hdt: HandDetector, bgr: BgRemover, img_size=28) -> Union[np.ndarray, None]:
    """
    Background-Norm

    *Hand-Detection-Filter

    ROI-Norm

    Illumination-Norm

    Channel-Norm

    Resolution-Norm

    :param image:
    :param hdt:
    :param bgr:
    :param img_size:
    :return:
    """
    # load image
    image = get_img_ndarray(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to load image"
        print(msg)
        return

    # process image
    image = bg_normalize(image, bgr)

    if not has_single_hand(image, hdt):
        msg = f"[PIPE-WARN] - failed to pass t_pipeline_a. By: can't detect any hand after bg_remove"
        print(msg)
        return

    image = roi_normalize(image, hdt)
    if image is None:
        msg = f"[PIPE-WARN] - failed to pass t_pipeline_a. By: failed to get after bg_remove"
        print(msg)
        return

    image = illumination_normalize(image)

    image = channel_normalize(image)

    image = resolution_normalize(image, img_size)

    return image


def t_pipeline_a_demo(image_path: str, hdt: HandDetector, bgr: BgRemover,
                      img_size=28, s_dir='pipeline-demo') -> Union[np.ndarray, None]:
    """
    Background-Norm

    *Hand-Detection-Filter

    ROI-Norm

    Illumination-Norm

    Channel-Norm

    Resolution-Norm

    :param s_dir:
    :param image_path:
    :param hdt:
    :param bgr:
    :param img_size:
    :return:
    """
    pipe_name = 't-pipeline'
    image_name = image_path.split('/')[-1].split('.')[0]

    # load image
    phase_idx = 0
    phase = 'raw'
    image_raw = get_img_ndarray(image_path)
    if image_raw is None:
        msg = f"[PIPE-WARN] - (1) - failed to pass pipeline_base. By: failed to load image"
        print(msg)
        return
    image_sp = f"{s_dir}/{pipe_name}-{image_name}-phase{phase_idx}-{phase}.jpg"
    cv2.imwrite(image_sp, image_raw)

    # bg norm
    phase_idx = 1
    phase = 'bg-norm'
    image = bg_normalize(image_raw, bgr)
    if not has_single_hand(image, hdt):
        msg = f"[PIPE-WARN] - (3) - failed to pass pipeline_base. By: can't detect any hand in the image, after bg_norm"
        print(msg)
        return
    image_sp = f"{s_dir}/{pipe_name}-{image_name}-phase{phase_idx}-{phase}.jpg"
    cv2.imwrite(image_sp, image)

    if not has_single_hand(image, hdt):
        msg = f"[PIPE-WARN] - failed to pass t_pipeline_a. By: can't detect any hand after bg_remove"
        print(msg)
        return

    # roi norm
    phase_idx = 2
    phase = 'roi-norm'
    image = roi_normalize(image, hdt)
    if image is None:
        msg = f"[PIPE-WARN] - failed to pass t_pipeline_a. By: failed to get after bg_remove"
        print(msg)
        return
    image_sp = f"{s_dir}/{pipe_name}-{image_name}-phase{phase_idx}-{phase}.jpg"
    cv2.imwrite(image_sp, image)

    # skin norm
    phase_idx = 3
    phase = 'skin-norm'
    image = illumination_normalize(image)
    image_sp = f"{s_dir}/{pipe_name}-{image_name}-phase{phase_idx}-{phase}.jpg"
    cv2.imwrite(image_sp, image)

    # channel norm
    phase_idx = 4
    phase = 'channel-norm'
    image = channel_normalize(image)
    image_sp = f"{s_dir}/{pipe_name}-{image_name}-phase{phase_idx}-{phase}.jpg"
    cv2.imwrite(image_sp, image)

    # size norm
    phase_idx = 5
    phase = 'size-norm'
    norm_hand = resolution_normalize(image, img_size)
    image_sp = f"{s_dir}/{pipe_name}-{image_name}-phase{phase_idx}-{phase}.jpg"
    cv2.imwrite(image_sp, norm_hand)

    return norm_hand


def t_pipeline_with_da_1(image: Union[np.ndarray, str], hdt: HandDetector, bgr: BgRemover, img_size=28):
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
        aug_img = t_pipeline_a(img, hdt, bgr)
        if aug_img is not None:
            show_img(aug_img)
        # img_output_ls.append()

    # for a, b in zip(img_original_ls, img_output_ls):
    #     show_img(a)
    #     show_img(b)


def t_pipeline_with_da_2(image: Union[np.ndarray, str], hdt: HandDetector,
                         bgr: BgRemover, img_size=28) -> Union[list, None]:
    img_base_with_deg = list()
    img_output_ls = list()

    image = get_img_ndarray(image)
    if image is None:
        msg = f"[WARN] - failed to pass pipeline_1. By: failed to load image"
        print(msg)
        return

    base_image = roi_normalize(image, hdt)
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
