# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/11/23
"""
import time
from cv2 import cv2
import imutils
import numpy as np
from imutils.video import WebcamVideoStream
from imutils.paths import list_images
from asl_translater import *
from image_pipeline.preprocessing.hand_detection import draw_single_hand_roi
from asl_model import load_exist_model
from webcam_dev_tool import *
from image_pipeline import *
import tensorflow as tf
from typing import Union
import os


def update_screen(frame: np.ndarray, window_name=''):
    cv2.imshow(window_name, frame)


def get_norm_hand_in_frame(hand_roi: np.ndarray,
                           hdt: HandDetector, bgr: BgRemover) -> Union[np.ndarray, None]:
    norm_hand = pipeline_app(hand_roi, hdt, bgr, img_size=28)
    if norm_hand is None:
        return

    # show_img(norm_hand, 'norm-hand')
    save_hand(norm_hand, prefix='norm-hand')

    return norm_hand


def detect_hand_in_frame(frame: np.ndarray, padding=30) -> tuple:
    """
    :param padding:
    :param frame:

    - Success
        :return: True, np.ndarray, np.ndarray

    - Failed:
        :return: False, None, None
    """

    roi_annotated_frame = draw_single_hand_roi(frame, hdt)

    if roi_annotated_frame is None:
        return False, None, None

    roi_annotated_frame = cv2.flip(roi_annotated_frame, 1)

    hand_roi = fetch_single_hand_roi(frame, hdt, padding=60)
    # save_hand_roi(hand_roi)
    # show_img(hand_roi, 'hand_roi')

    return True, roi_annotated_frame, hand_roi


def translate(norm_hand: np.ndarray):
    # show_img(norm_hand, str(norm_hand.shape))
    norm_hand = np.expand_dims(norm_hand, 0)
    print(norm_hand.shape)



def get_display_frame_from_raw(raw_frame: np.ndarray):
    return cv2.flip(raw_frame, 1)


def main():
    flag = True
    vs = WebcamVideoStream(src=0)
    vs.start()
    time.sleep(2.0)

    cv2.startWindowThread()
    while flag:

        frame = vs.read()
        if frame is None:
            continue

        frame = imutils.resize(frame, width=800)  # save this for prediction
        display_frame = get_display_frame_from_raw(frame)

        norm_hand = None
        detect_result, hand_roi_annotated_frame, hand_roi = detect_hand_in_frame(frame)
        if detect_result:
            display_frame = hand_roi_annotated_frame
            update_screen(display_frame, M_SCREEN_NAME)

            norm_hand = get_norm_hand_in_frame(hand_roi, hdt, bgr)

        if norm_hand is not None:
            translate(norm_hand)

        text = 'display frame'
        display_frame = add_text_in_frame(display_frame, text)
        # save_frame(frame)
        # save_display_frame(display_frame)
        update_screen(display_frame, M_SCREEN_NAME)

        key = cv2.waitKey(1)

        if key == 27 or key == ord('q'):
            flag = False
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            cv2.waitKey(500)

    vs.stop()
    cv2.destroyAllWindows()


def load_aslt_model(model_dir='./ASLT-Model/app-model') -> tf.keras.Sequential:
    return load_exist_model(model_dir)


if __name__ == '__main__':
    M_SCREEN_NAME = 'ASL Translator'

    # preload ASLT app model
    check_model_exist()
    model = load_aslt_model()
    msg = f"[INFO] - ASLT-Model loaded"
    print(msg)
    # model.summary()

    # preload detectors
    msg = f"[INFO] - Loading detectors..."
    print(msg)
    hdt = HandDetector()
    bgr = BgRemover()
    bgr.load_model()
    msg = f"[INFO] - Detectors loaded"
    print(msg)

    remove_old_images()
    main()
