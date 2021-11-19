"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/18/21
"""
# coding: utf-8
from image_pipeline import *
from cv2 import cv2
import numpy as np
from image_pipeline import pipe_1

# image_path = 'img_for_dev/A1.jpg'
image_path = 'img_for_dev/peace.jpg'


def show_img(image: np.ndarray, widow_name=''):
    cv2.imshow(widow_name, image)
    cv2.waitKey(0)


if __name__ == '__main__':
    # image_raw = cv2.imread(image_path)
    pipe_1(image_path)
