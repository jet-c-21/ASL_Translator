import cv2
import numpy as np
import mediapipe as mp
import imutils
from numpy.lib.type_check import imag
from skimage.util import random_noise


def fetch_hand_roi(image, padding=15, bg=False, which_hand='Right'):
    """
    fetch_hand_roi(numpy array, int, bool) -> numpy array, numpy array
    takes a numpy array representing a image and an integer representing padding.

    @param: image is a numpy array
    @param: padding is a integer value representing the padding of the cropped image
    @param: black out the rest of the image outside of region of interest
    return_0: whole image with rectange repr region of interest
    return_1: extracted region of interest

    """

    num_of_hands = 2

    # Handedness is mirrored for mediapipe, this adjusts for this
    if which_hand == 'Right':
        which_hand = 'Left'
        num_of_hands = 1
    elif which_hand == 'Left':
        which_hand = 'Right'
        num_of_hands = 1

    mphands = mp.solutions.hands
    hands = mphands.Hands(max_num_hands=num_of_hands)

    h, w, c = image.shape

    black_bg = np.zeros_like(image)

    cropped_img = None

    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(imagergb)
    hand_landmarks = result.multi_hand_landmarks
    handedness = result.multi_handedness

    # find which_hand
    if hand_landmarks:
        if which_hand != 'Both':
            for _, handLMs in enumerate(hand_landmarks):
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm in handLMs.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    cv2.rectangle(image, (x_min - padding, y_min - padding),
                                  (x_max + padding, y_max + padding), (0, 255, 0), 2)
                    cropped_img = image[(
                        y_min - padding):(y_max + padding), (x_min - padding):(x_max + padding)]
        else:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                cv2.rectangle(image, (x_min - padding, y_min - padding),
                              (x_max + padding, y_max + padding), (0, 255, 0), 2)
                # cropped_img = image[(y_min - padding):(y_max + padding), (x_min - padding):(x_max + padding)]

    if bg and which_hand != 'Both':
        if cropped_img is None:
            image = black_bg
        else:
            black_bg[(y_min - padding):(y_max + padding),
                     (x_min - padding):(x_max + padding)] = cropped_img
            image = black_bg

    return image, cropped_img


def rgb_to_hsv(image, s=32, m=128):
    """
    rgb_to_hsv(numpy array, int, int) -> (numpy array, numpy array, numpy array)
    takes a numpy array repr. an image and integers representing arbitrary values for averaging brightness processing

    @param: images is a numpy array
    @param: standard deviation is set by arbitrary value s
    @param: brightness average is set by adding an arbitrary value m
    returns tuple of a smooth image, adaptive smooth image, and a averaged brightness

    """
    hsv = cv2.cvtColor(
        image, cv2.COLOR_RGB2HSV)  # Convert RGB to hsv color system (hue, saturation, and value(brightness))
    h, s, v = cv2.split(hsv)

    ####

    flattening_result = cv2.equalizeHist(v)

    smooth_image = cv2.cvtColor(
        cv2.merge((h, s, flattening_result)), cv2.COLOR_HSV2BGR)

    ####

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))

    adaptive_flattening_result = clahe.apply(v)

    adaptive_smooth_image = cv2.cvtColor(
        cv2.merge((h, s, adaptive_flattening_result)), cv2.COLOR_HSV2BGR)

    ####

    v = (v-np.mean(v)) / np.std(v) * s + m

    averaging_result = np.array(v, dtype=np.uint8)

    averaging_brightness_image = cv2.cvtColor(
        cv2.merge((h, s, averaging_result)), cv2.COLOR_HSV2BGR)

    return smooth_image, adaptive_smooth_image, averaging_brightness_image


def bg_normalization_red_channel(image, color=0):
    """
    bg_normalization(numpy array, int) -> numpy array
    Takes an image

    """

    # Extract red color channel
    gray = image[:, :, 2]

    # Apply binary threshold using automatically selected threshold (using cv2.THRESH_OTSU parameter).
    ret, thresh_gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use "opening" morphological operation for clearing some small dots (noise)
    thresh_gray = cv2.morphologyEx(
        thresh_gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Use "closing" morphological operation for closing small gaps
    thresh_gray = cv2.morphologyEx(
        thresh_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # Invert Image to Obtain Mask
    mask = cv2.bitwise_not(thresh_gray)

    # Load Background
    background = np.full(image.shape, color, dtype=np.uint8)

    # Masked Foreground (We can stop here if all you want is a black background)
    foreground_mask = cv2.bitwise_and(image, image, mask=mask)

    # Get Masked Background, mask must be inverted
    inverted_mask = cv2.bitwise_not(mask)
    background_mask = cv2.bitwise_and(
        background, background, mask=inverted_mask)

    # Combine Masked Background and Masked Foreground
    final = cv2.bitwise_or(foreground_mask, background_mask)

    return final


def bg_normalization_fg_extraction(image, min=.0005):

    def get_contours(image):
        """
        This function finds all the contours in an image and return the largest
        contour area.
        :param image: a binary image
        """
        canny_low = 15
        canny_high = 150

        image = cv2.Canny(image, canny_low, canny_high)

        # show('inter', image)

        image = cv2.dilate(image, None)
        image = cv2.erode(image, None)
        image = image.astype(np.uint8)

        # show('inter', image)

        contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

        return contour_info

    def show(name, image):
        """
        A simple function to visualize OpenCV images on screen.
        :param name: a string signifying the imshow() window name
        :param image: NumPy image to show
        """
        cv2.imshow(name, image)
        cv2.waitKey(0)

    # Parameters
    min_area = min
    max_area = .95
    dilate_iter = 10
    erode_iter = 10
    blur = 21

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show('intermediate', image)

    image_area = image.shape[0] * image.shape[1]

    max_area = max_area * image_area
    min_area = min_area * image_area

    # Setting up mask with a matrix of 0's
    mask = np.zeros_like(image_gray)

    # Get contours and their areas
    contour_info = get_contours(image_gray)

    # Add relevant contours to mask
    for contour in contour_info:
        if contour[1] > min_area and contour[1] < max_area:
            mask = cv2.fillConvexPoly(mask, contour[0], 255)
            # show('inter', mask)

    # show('inter', mask)

    # Create copy of current mask
    res_mask = np.copy(mask)
    res_mask[mask == 0] = cv2.GC_BGD
    res_mask[mask == 255] = cv2.GC_PR_BGD
    res_mask[mask == 255] = cv2.GC_FGD

    mask2 = np.where((res_mask == cv2.GC_FGD) | (
        res_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    new_mask3d = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
    mask3d = new_mask3d
    mask3d[new_mask3d > 0] = 255.0
    mask3d[mask3d > 255] = 255.0
    # apply Gaussian blurring to smoothen out the edges a bit
    # `mask3d` is the final foreground mask (not extracted foreground image)
    mask3d = cv2.GaussianBlur(mask3d, (5, 5), 0)

    # create the foreground image by zeroing out the pixels where `mask2`...
    # ... has black pixels
    foreground = np.copy(image).astype(float)
    foreground[mask3d == 0] = 0
    foreground = foreground.astype(np.uint8)

    return foreground


def da_rotate(image, angle):
    return imutils.rotate_bound(image, angle)


def da_flip(image, orientation=0):
    # 0: vertical
    # 1: horizontal
    return cv2.flip(image, orientation)


def da_add_noise(image, type='s&p'):
    if type == "s&p":
        noisey_image = random_noise(image, mode=type, amount=0.3)
        noisey_image = np.array(255*noisey_image, dtype='uint8')
        return noisey_image
    elif type == 'speckle':
        noisey_image = random_noise(image, mode=type)
        noisey_image = np.array(255*noisey_image, dtype='uint8')
        return noisey_image
    elif type == 'gaussian':
        noisey_image = random_noise(image, mode=type, mean=.10)
        noisey_image = np.array(255*noisey_image, dtype='uint8')
        return noisey_image
    elif type == 'poisson':
        noisey_image = random_noise(image, mode=type)
        noisey_image = np.array(255*noisey_image, dtype='uint8')
        return noisey_image



def da_filter(image, type='blur', amount=7):
    if type == 'blur':
        return cv2.medianBlur(image, amount)
    elif type == 'laplacian':
        return cv2.Laplacian(image,cv2.CV_64F)
    elif type == 'sobelx':
        return cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    elif type == 'sobely':
        return cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)


def da_dilation(image, iter=1):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(image, kernel=kernel, iterations=iter)


def da_erosion(image, iter=1):
    kernel = np.ones((5,5), np.uint8)
    return cv2.erode(image, kernel=kernel, iterations=iter)


def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize(image, scale):
    dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
