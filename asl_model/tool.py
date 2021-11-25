# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/22/21
"""
from .models import get_model_1, get_model_2, get_model_3
from .stl import *

all_models = {
    get_model_1.__name__: get_model_1,
    get_model_2.__name__: get_model_2,
    get_model_3.__name__: get_model_3,
    get_stn_a_model_1.__name__: get_stn_a_model_1,
    get_stn_a_model_2.__name__: get_stn_a_model_2,
    get_stn_a_model_3.__name__: get_stn_a_model_3,
    get_stn_a_model_4.__name__: get_stn_a_model_4,
    get_stn_a_model_5.__name__: get_stn_a_model_5,
    get_stn_a_model_6.__name__: get_stn_a_model_6,
    get_stn_a_model_7.__name__: get_stn_a_model_7,
    get_stn_a_model_8.__name__: get_stn_a_model_8,
    get_stn_a_model_9.__name__: get_stn_a_model_9,
}

def load_exist_model(model_dir_path: str):
    """
    Notes: model_dir_path is the path to the entire model folder, not checkpoint

    :param model_dir_path:
    :return:
    """

    return tf.keras.models.load_model(model_dir_path)