# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""
import numpy as np
from .cls.bg_remover import BgRemover
from typing import Union


def remove_bg(image: np.ndarray, bgr: BgRemover) -> Union[np.ndarray, None]:
    cutout = bgr.remove_bg(image)
    if cutout is None:
        msg = f"[WARN] - failed to get cutout by BgRemover"
        print(msg)
        return

    return cutout
