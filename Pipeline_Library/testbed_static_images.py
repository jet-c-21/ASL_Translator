import Pipeline as pl
import cv2

image = cv2.imread('peace_3.jpg')

cropped_image = pl.fetch_hand_roi(image, 50)

cv2.imwrite("cropped_image.jpg", cropped_image)

cv2.imshow("Frame", cropped_image)
cv2.waitKey(1)



