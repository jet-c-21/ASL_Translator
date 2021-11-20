# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""
import time
from image_pipeline import *
from image_pipeline.preprocessing import *

if __name__ == '__main__':
    img_path = 'img_for_dev/peace_0.jpg'

    # s = time.time()
    # norm_hand_0 = pipeline_0(img_path)
    # e = time.time()
    # show_img(norm_hand_0, f"cost = {e - s}")

    bgr = BgRemover()
    bgr.load_model()

    norm_hand_1 = pipeline_1(img_path, bgr)

    norm_hand_2 = pipeline_2(img_path, bgr)

    img_plt_save(norm_hand_2)
