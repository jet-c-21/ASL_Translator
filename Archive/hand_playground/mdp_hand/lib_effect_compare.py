"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/18/21
"""
# coding: utf-8
from cv2 import cv2
import numpy as np
from imutils.convenience import rotate_bound


def show_image(image: np.ndarray):
    cv2.imshow('', image)
    cv2.waitKey(0)




if __name__ == '__main__':
    IMG_PATH = 'hand.jpg'
    img_raw = cv2.imread(IMG_PATH)

    ir = rotate_bound(img_raw, 30)
    show_image(ir)
    show_image(img_raw)
