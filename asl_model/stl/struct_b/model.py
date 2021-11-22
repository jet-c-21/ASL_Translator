# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/22/21
"""
from .ult import get_stn
from tensorflow.keras import layers, models, initializers
import tensorflow as tf

STN_SEED = 777


# def get_stn_b_model_0(img_size=28):
#     input_layers = layers.Input((img_size, img_size, 1))
#     x = stn(input_layers)
#
#     x = layers.Conv2D(64, [9, 9], activation='relu')(x)
#     x = layers.MaxPool2D()(x)
#     x = layers.Conv2D(64, [7, 7], activation='relu')(x)
#     x = layers.MaxPool2D()(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(64, activation='relu')(x)
#     x = layers.Dense(32, activation='relu')(x)
#     x = layers.Dense(26, activation='softmax')(x)
#
#     return models.Model(inputs=input_layers, outputs=x)

def get_stn_b_model_1(img_size=28):
    input_layers = layers.Input((img_size, img_size, 1))
    x = get_stn(input_layers)

    x = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    output_layers = layers.Dense(26, activation="softmax")(x)

    return models.Model(inputs=input_layers, outputs=output_layers)
