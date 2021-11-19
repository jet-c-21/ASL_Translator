import Pipeline as pl
import cv2

image = cv2.imread('peace_1.jpg')

image_with_roi, roi = pl.fetch_hand_roi(image, 50)

# cv2.imwrite("cropped_image.jpg", cropped_image)

cv2.imshow("Frame", roi)
cv2.waitKey(0)
