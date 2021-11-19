"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/18/21
"""
# coding: utf-8
from image_pipeline import *
from cv2 import cv2
import numpy as np

image_path = 'img_for_dev/A1.jpg'


def show_image(image: np.ndarray):
    cv2.imshow('', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    image_raw = cv2.imread(image_path)
    hand_roi = fetch_single_hand_roi(image_raw)

    hr_gray = grayscale(hand_roi)
    show_image(hr_gray)
