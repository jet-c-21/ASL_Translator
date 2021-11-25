"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""
# coding: utf-8
import time

import PIL
import numpy as np
from cv2 import cv2
from rembg.bg import get_model, naive_cutout
from rembg.u2net.detect import predict


def pil_image_to_np_ndarray(image: PIL.Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def np_ndarray_to_pil_image(image: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def show_img(image: np.ndarray):
    cv2.imshow('', image)
    cv2.waitKey(0)


def remove_bg(model, image: np.ndarray) -> np.ndarray:
    # image = np_ndarray_to_pil_image(image)
    mask = predict(model, image).convert("L")
    # mask.show()

    image = np_ndarray_to_pil_image(image)
    cutout = naive_cutout(image, mask)
    print(f"cutout type = {type(cutout)}")

    return pil_image_to_np_ndarray(cutout)


IMG_PATH = 'peace_0.jpg'

if __name__ == '__main__':
    img = cv2.imread(IMG_PATH)
    # show_img(img)

    u2net_model = get_model('u2net')

    s = time.time()
    res = remove_bg(u2net_model, img)
    e = time.time()

    print(e - s)

    show_img(res)
