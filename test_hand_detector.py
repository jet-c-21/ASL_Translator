# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/21/21
"""
from image_pipeline.preprocessing import HandDetector
from image_pipeline.preprocessing.hand_detection import detect_single_hand, fetch_single_hand_roi
from image_pipeline import get_img_ndarray, show_img

if __name__ == '__main__':
    IMG_PATH = 'img_for_dev/peace_0.jpg'
    image = get_img_ndarray(IMG_PATH)
    hdt = HandDetector()

    hand_landmarks = detect_single_hand(image, hdt)
    print(hand_landmarks)

    roi = fetch_single_hand_roi(image, hdt)
    show_img(roi, 'roi')
