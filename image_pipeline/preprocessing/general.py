"""
Author: Eugene Mondkar
GitHub: https://github.com/EugeneMondkar
Create Date: 11/17/21
"""

# coding: utf-8
from cv2 import cv2
import numpy as np
import imutils
import mediapipe as mdp
from skimage.util import random_noise


def rgb_to_hsv(image: np.ndarray, s=32, m=128) -> (np.ndarray, np.ndarray, np.ndarray):
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

    v = (v - np.mean(v)) / np.std(v) * s + m

    averaging_result = np.array(v, dtype=np.uint8)

    averaging_brightness_image = cv2.cvtColor(
        cv2.merge((h, s, averaging_result)), cv2.COLOR_HSV2BGR)

    return smooth_image, adaptive_smooth_image, averaging_brightness_image


def bg_normalization_red_channel(image: np.ndarray, color=0) -> np.ndarray:
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
    background_mask = cv2.bitwise_and(background, background, mask=inverted_mask)

    # Combine Masked Background and Masked Foreground
    final = cv2.bitwise_or(foreground_mask, background_mask)

    return final


def get_contours(image: np.ndarray):
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


def bg_normalization_fg_extraction(image, min_area=.0005) -> np.ndarray:
    # Parameters
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
        if min_area < contour[1] < max_area:
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


def da_rotate(image: np.ndarray, angle: int) -> np.ndarray:
    return imutils.rotate_bound(image, angle)


def da_flip(image: np.ndarray, orientation=0) -> np.ndarray:
    # 0: vertical
    # 1: horizontal
    return cv2.flip(image, orientation)


def da_add_noise(image, mode='s&p'):
    if mode == "s&p":
        noised_image = random_noise(image, mode=mode, amount=0.3)
        noised_image = np.array(255 * noised_image, dtype='uint8')
        return noised_image

    elif mode == 'speckle':
        noised_image = random_noise(image, mode=mode)
        noised_image = np.array(255 * noised_image, dtype='uint8')
        return noised_image

    elif mode == 'gaussian':
        noised_image = random_noise(image, mode=mode, mean=.10)
        noised_image = np.array(255 * noised_image, dtype='uint8')
        return noised_image

    elif mode == 'poisson':
        noised_image = random_noise(image, mode=mode)
        noised_image = np.array(255 * noised_image, dtype='uint8')
        return noised_image


def da_filter(image: np.ndarray, mode='blur', k_size=5) -> np.ndarray:
    if mode == 'blur':
        return cv2.medianBlur(image, k_size)

    elif mode == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F)

    elif mode == 'sobelx':
        return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k_size)

    elif mode == 'sobely':
        return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k_size)


def da_dilation(image: np.ndarray, k_size=5, iterations=1) -> np.ndarray:
    kernel = np.ones((k_size, k_size), np.uint8)
    return cv2.dilate(image, kernel=kernel, iterations=iterations)


def da_erosion(image: np.ndarray, k_size=5, iterations=1) -> np.ndarray:
    kernel = np.ones((k_size, k_size), np.uint8)
    return cv2.erode(image, kernel=kernel, iterations=iterations)


def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_by_scale(image: np.ndarray, scale) -> np.ndarray:
    dim_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return cv2.resize(image, dim_size, interpolation=cv2.INTER_AREA)


def resize(image: np.ndarray, img_size) -> np.ndarray:
    return cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
