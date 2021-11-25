# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/11/23

Based on work by Eugene Mondkar
GitHub: https://github.com/EugeneMondkar
Refer to Archive/Pipeline_Library/testbed_cam.py
"""
import time
from cv2 import cv2
import imutils
import numpy as np
from imutils.video import WebcamVideoStream
from asl_translater import add_text_in_frame
from image_pipeline.preprocessing import gen_random_token

if __name__ == '__main__':
    flag = True
    vs = WebcamVideoStream(src=0)
    vs.start()
    time.sleep(2.0)
    fetch_face = 0
    last_face_ts = 0
    cv2.startWindowThread()
    while flag:
        frame = vs.read()
        if frame is None:
            continue
        frame = imutils.resize(frame, width=800)

        display_frame = frame.copy()

        display_frame = cv2.flip(display_frame, 1)

        text = 'Web Cam Test'

        display_frame = add_text_in_frame(display_frame, text, coord=(0, 50))

        text = 'press q to quit'
        display_frame = add_text_in_frame(display_frame, text, coord=(70, 120))

        cv2.imshow('Frame', display_frame)

        key = cv2.waitKey(1)

        if key == 27 or key == ord('q'):
            flag = False
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            cv2.waitKey(500)

    vs.stop()
    cv2.destroyAllWindows()
