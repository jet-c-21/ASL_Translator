# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/21/21
"""
from imutils.paths import list_images
from data_tool import *

if __name__ == '__main__':
    OUTPUT_DIR_ROOT = 'TEST_DATASET_A'
    OUTPUT_AP_DIR_ROOT = 'TEST_DATASET_A_AP'

    DATASET2_DIR_PATH = 'dataset3'

    x = list(list_images(DATASET2_DIR_PATH))
    print(len(x))
