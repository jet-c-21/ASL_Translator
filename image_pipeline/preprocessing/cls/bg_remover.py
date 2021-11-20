# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""

from typing import Union

import numpy as np
from rembg.bg import get_model, naive_cutout
from rembg.u2net.detect import predict

from image_pipeline.preprocessing.ult import np_ndarray_to_pil_image, pil_image_to_np_ndarray


class BgRemover:
    mask_model = None

    def __init__(self):
        pass

    def load_model(self, model_name='u2net'):
        if self.mask_model is None:
            BgRemover.mask_model = get_model(model_name)
        msg = '[INFO] - model loaded'
        print(msg)

    def remove_bg(self, image: np.ndarray) -> Union[np.ndarray, None]:
        if self.mask_model is None:
            msg = f"[WARN] - model unloaded, please use load_model()"
            print(msg)
            return

        mask = predict(self.mask_model, image).convert('L')
        if mask is None:
            msg = f"[WARN] - failed to predict mask by mask_model"
            print(msg)
            return

        image = np_ndarray_to_pil_image(image)
        cutout = naive_cutout(image, mask)

        return pil_image_to_np_ndarray(cutout)
