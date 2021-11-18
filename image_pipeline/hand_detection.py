"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/18/21
"""
# coding: utf-8
from typing import Union

import mediapipe as mdp
import numpy as np

from cv2 import cv2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


def detect_single_hand(image: np.ndarray) -> Union[NormalizedLandmarkList, None]:
    mdp_hands = mdp.solutions.hands

    hand_detector = mdp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    image = cv2.flip(image, 1)
    detect_result = hand_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # prevent to change the image color

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


def get_hand_roi_coord(image: np.ndarray, hand_landmarks: NormalizedLandmarkList):
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


def draw_single_hand_roi(image: np.ndarray, padding=15) -> Union[np.ndarray, None]:
    hand_landmarks = detect_single_hand(image)
    if not hand_landmarks:
        msg = f"[WARN] - Failed to draw single hand roi."
        print(msg)
        return

    x_min, x_max, y_min, y_max = get_hand_roi_coord(image, hand_landmarks)

    annotated_img = image.copy()
    cv2.rectangle(annotated_img,
                  (x_min - padding, y_min - padding),
                  (x_max + padding, y_max + padding),
                  (0, 255, 0), 2)

    return annotated_img


def fetch_single_hand_roi(image: np.ndarray, padding=15) -> Union[np.ndarray, None]:
    hand_landmarks = detect_single_hand(image)
    if not hand_landmarks:
        msg = f"[WARN] - Failed to fetch single hand roi."
        print(msg)
        return

    x_min, x_max, y_min, y_max = get_hand_roi_coord(image, hand_landmarks)
    roi = image[(y_min - padding):(y_max + padding), (x_min - padding):(x_max + padding)]
    return roi
