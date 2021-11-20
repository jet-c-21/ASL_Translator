"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/18/21
"""
# coding: utf-8
from image_pipeline import *
from cv2 import cv2
import numpy as np
from image_pipeline.preprocessing import img_show

# image_path = 'img_for_dev/A1.jpg'
image_path = 'img_for_dev/peace_0.jpg'


def show_img(image: np.ndarray, widow_name=''):
    cv2.imshow(widow_name, image)
    cv2.waitKey(0)


if __name__ == '__main__':
    # image_raw = cv2.imread(image_path)
    bgr = BgRemover()
    bgr.load_model()

    aug_img_ls = t_pipeline_with_da_2(image_path, bgr)
    for i in aug_img_ls:
        show_img(i)
    # show_img(norm_hand)
    # print(norm_hand.shape)
