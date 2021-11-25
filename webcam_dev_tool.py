# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/11/23
"""
import os
from imutils.paths import list_images
from image_pipeline.preprocessing import gen_random_token
from cv2 import cv2
import numpy as np

img_dir = 'img_from_cam'


def remove_old_images():
    for p in list_images(img_dir):
        os.remove(p)


def save_frame(img: np.ndarray, rtk_len=3):
    img_path = f"{img_dir}/frame/{gen_random_token(rtk_len)}.jpg"
    cv2.imwrite(img_path, img)
    print(f"frame: {img_path} saved")


def save_display_frame(img: np.ndarray, rtk_len=3):
    img_path = f"{img_dir}/display/{gen_random_token(rtk_len)}.jpg"
    cv2.imwrite(img_path, img)
    print(f"display frame: {img_path} saved")


def save_roi(img: np.ndarray, rtk_len=3, prefix=''):
    if prefix:
        img_path = f"{img_dir}/hand_roi/{prefix}_{gen_random_token(rtk_len)}.jpg"
    else:
        img_path = f"{img_dir}/hand_roi/{gen_random_token(rtk_len)}.jpg"

    cv2.imwrite(img_path, img)
    print(f"[INFO] - Image : {img_path} saved")


def save_alphabet(norm_hand: np.ndarray, alphabet: str, rtk_len=6):
    img_path = f"{img_dir}/pred/{alphabet}-{gen_random_token(rtk_len)}.jpg"
    cv2.imwrite(img_path, norm_hand)
    print(f"[INFO] - Alphabet : {img_path} saved")


def save_hand(hand: dict, rtk_len=3):
    roi = hand['roi']
    norm_hand = hand['norm']
    alphabet = hand['alphabet']
    rt = gen_random_token(rtk_len)

    roi_path = f"{img_dir}/pred/{alphabet}-roi-{rt}.jpg"
    norm_path = f"{img_dir}/pred/{alphabet}-norm-{rt}.jpg"

    cv2.imwrite(roi_path, roi)
    cv2.imwrite(norm_path, norm_hand)
