"""
author: Eugene Mondkar
GitHub: https://github.com/EugeneMondkar
Create Date: 11/17/21
"""

import Pipeline as pl
import cv2

cap = cv2.VideoCapture(0)

# model = load_model()

while(True):
    _, frame = cap.read()

    frame, roi = pl.fetch_hand_roi(frame, 50, bg=True, which_hand='Left')

    cv2.imshow("Frame", frame)

    if roi is not None:
        try:
            # roi = pl.da_rotate(roi, 30)
            # roi = pl.bg_normalization_fg_extraction(roi)
            # roi = pl.da_flip(roi, 1)
            roi = pl.da_add_noise(roi, 'gaussian')
            # roi = pl.da_filter(roi, 'laplacian')
            roi = pl.da_dilation(roi, 1)
            # roi = pl.da_erosion(roi, 1)
            roi = pl.greyscale(roi)
            roi = pl.resize(roi, 1.5)
            cv2.imshow("Region of Interest", roi)
        except:
            pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
