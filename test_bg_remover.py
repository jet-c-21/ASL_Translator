"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""
# coding: utf-8
import time
from cv2 import cv2
from image_pipeline.preprocessing import BgRemover, remove_bg, show_img
from image_pipeline.preprocessing import da_add_noise, da_filter, da_dilation, da_erosion

if __name__ == '__main__':
    img_path = 'img_for_dev/peace_0.jpg'
    image = cv2.imread(img_path)
    # image = da_add_noise(image, 'poisson')
    # image = da_filter(image, 'blur')
    # image = da_dilation(image, iterations=1)
    image = da_erosion(image, iterations=1)

    bgr = BgRemover()
    bgr.load_model()

    img_bg_rm = remove_bg(image, bgr)

    # img_bg_rm = da_add_noise(img_bg_rm, 'speckle')

    show_img(img_bg_rm)
