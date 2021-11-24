# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/23/21
"""
import os
from imutils.paths import list_images
from data_tool import create_dir
from image_pipeline import HandDetector, BgRemover, pipeline_base_demo, t_pipeline_a_demo

save_dir = 'pipeline-demo'
create_dir(save_dir)


def remove_old_images():
    for p in list_images(save_dir):
        os.remove(p)


if __name__ == '__main__':
    # img_path = 'DATASET_A/train/Y/TRAIN_Y_78.jpg'
    img_path = 'DATASET_A/train/F/TRAIN_F_31.jpg'
    hdt = HandDetector()
    bgr = BgRemover()
    bgr.load_model()

    remove_old_images()

    pipeline_base_demo(img_path, hdt, bgr, s_dir=save_dir)
    t_pipeline_a_demo(img_path, hdt, bgr, s_dir=save_dir)
