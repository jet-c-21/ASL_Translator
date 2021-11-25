# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 10/23/21
"""
from tensorflow.keras import layers

img_size = 28
input_layer = layers.Input(shape=(img_size, img_size, 1))

print(input_layer.shape)
