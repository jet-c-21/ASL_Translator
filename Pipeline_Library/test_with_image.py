import Pipeline as pl
from cv2 import cv2
import numpy as np


def show_img(image: np.ndarray):
    cv2.imshow('test', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    IMG_PATH = 'peace.jpg'
    image_raw = cv2.imread(IMG_PATH)

    cropped_image = pl.fetch_hand_roi(image_raw, 50)
    show_img(cropped_image)
    # print(cropped_image)

# cv2.imwrite("cropped_image.jpg", cropped_image)
# cv2.imshow("Frame", cropped_image)
# cv2.waitKey(1)