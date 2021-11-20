"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""
# coding: utf-8
from image_pipeline.preprocessing.hand_detection import *

from cv2 import cv2


def show_img(image: np.ndarray, widow_name=''):
    cv2.imshow(widow_name, image)
    cv2.waitKey(0)


if __name__ == '__main__':
    image_path = 'img_for_dev/peace_0.jpg'
    image = cv2.imread(image_path)

    show_img(image, 'raw')

    roi = fetch_single_hand_roi(image)
    show_img(roi, 'roi')
    show_img(image, 'after')
