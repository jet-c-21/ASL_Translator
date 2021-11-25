# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/21/21
"""
import mediapipe

mp_hands = mediapipe.solutions.hands


class HandDetector(mp_hands.Hands):
    def __init__(self, mode='single'):
        self.mode = mode
        if self.mode == 'single':
            super().__init__(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        else:
            super().__init__(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
