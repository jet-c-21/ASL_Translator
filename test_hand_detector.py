# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/21/21
"""
from image_pipeline.preprocessing import HandDetector
from image_pipeline.preprocessing.hand_detection import draw_single_hand_roi, detect_single_hand, fetch_single_hand_roi
from image_pipeline import get_img_ndarray, show_img

if __name__ == '__main__':
    IMG_PATH = 'img_for_dev/peace_1.jpg'
    image = get_img_ndarray(IMG_PATH)
    hdt = HandDetector()

    noted_hand = draw_single_hand_roi(image, hdt)
    show_img(noted_hand, 'noted_hand')

    roi = fetch_single_hand_roi(image, hdt)
    show_img(roi, 'roi')
