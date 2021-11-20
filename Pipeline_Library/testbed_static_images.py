import Pipeline as pl
from cv2 import cv2
import time

image = cv2.imread('peace_1.jpg')

s = time.time()
image_with_roi, roi = pl.fetch_hand_roi(image, 50)
e = time.time()
print(f"cost = {e - s}")

roi = pl.bg_normalization_fg_extraction(roi)

cv2.imwrite("roi_bg_norm_issue.jpg", roi)

# cv2.imshow("Frame", roi)
# cv2.waitKey(0)
