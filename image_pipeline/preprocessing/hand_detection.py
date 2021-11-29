"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/18/21

Based on work by Eugene Mondkar
GitHub: https://github.com/EugeneMondkar
Refer to Archive/Pipeline_Library/Pipeline.py
"""
# coding: utf-8
from typing import Union

import mediapipe as mdp
import numpy as np

from cv2 import cv2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from .cls import HandDetector


def detect_single_hand(image: np.ndarray, hdt: HandDetector) -> Union[NormalizedLandmarkList, None]:
    image = cv2.flip(image, 1)

    detect_result = hdt.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # prevent to change the image color

    hand_landmarks_ls = detect_result.multi_hand_landmarks

    if not hand_landmarks_ls:
        msg = f"[WARN] - Failed to detect hand by hand_detector"
        print(msg)
        return

    if len(hand_landmarks_ls) != 1:
        msg = f"[WARN] - Detected hand count != 1, get: {len(hand_landmarks_ls)}"
        print(msg)
        return

    return hand_landmarks_ls[0]


def _get_hand_roi_coord(image: np.ndarray, hand_landmarks: NormalizedLandmarkList):
    image_height, image_width, _ = image.shape
    x_min = image_width
    y_min = image_height
    x_max, y_max = 0, 0

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * image_width), int(lm.y * image_height)

        if x > x_max:
            x_max = x

        if x < x_min:
            x_min = x

        if y > y_max:
            y_max = y

        if y < y_min:
            y_min = y

    return x_min, x_max, y_min, y_max


def draw_single_hand_roi(image: np.ndarray, hdt: HandDetector, padding=15) -> Union[np.ndarray, None]:
    hand_landmarks = detect_single_hand(image, hdt)
    if not hand_landmarks:
        msg = f"[WARN] - Failed to draw single hand roi."
        print(msg)
        return

    image = cv2.flip(image, 1)
    x_min, x_max, y_min, y_max = _get_hand_roi_coord(image, hand_landmarks)

    annotated_img = image.copy()
    cv2.rectangle(annotated_img,
                  (x_min - padding, y_min - padding),
                  (x_max + padding, y_max + padding),
                  (0, 255, 0), 2)

    annotated_img = cv2.flip(annotated_img, 1)

    return annotated_img


def fetch_single_hand_roi(image: np.ndarray, hdt: HandDetector, padding=15) -> Union[np.ndarray, None]:
    hand_landmarks = detect_single_hand(image, hdt)  # the landmark is getting from the flipped mode

    if not hand_landmarks:
        msg = f"[WARN] - Failed to fetch single hand roi."
        print(msg)
        return

    image = cv2.flip(image, 1)
    x_min, x_max, y_min, y_max = _get_hand_roi_coord(image, hand_landmarks)
    roi = image[(y_min - padding):(y_max + padding), (x_min - padding):(x_max + padding)]

    roi = cv2.flip(roi, 1)
    return roi


def has_single_hand(image: np.ndarray, hdt: HandDetector) -> bool:
    if detect_single_hand(image, hdt) is None:
        return False
    return True
