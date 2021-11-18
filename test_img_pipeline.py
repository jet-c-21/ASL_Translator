"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/18/21
"""
# coding: utf-8
from image_pipeline import fetch_hand_roi
from cv2 import cv2
import numpy as np

image_path = 'img_for_dev/A1.jpg'


def show_image(image: np.ndarray):
    cv2.imshow('', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    image_raw = cv2.imread(image_path)
    img1, img2 = fetch_hand_roi(image_raw, bg=1)

    show_image(img1)
    show_image(img2)
