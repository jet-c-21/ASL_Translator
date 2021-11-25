"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/18/21
"""
# coding: utf-8
from cv2 import cv2
import mediapipe as mdp

mdp_drawing = mdp.solutions.drawing_utils
mdp_drawing_styles = mdp.solutions.drawing_styles
mdp_hands = mdp.solutions.hands

IMAGE_FILES = []
with mdp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue

        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mdp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mdp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mdp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mdp_hands.HAND_CONNECTIONS,
                mdp_drawing_styles.get_default_hand_landmarks_style(),
                mdp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(
            '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mdp_drawing.plot_landmarks(
                hand_world_landmarks, mdp_hands.HAND_CONNECTIONS, azimuth=5)
