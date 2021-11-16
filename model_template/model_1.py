"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/16/21
"""
# coding: utf-8
import tensorflow as tf
from tensorflow.keras import layers, models


def get_model_1():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))

    # finish feature extraction
    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(25, activation='softmax'))

    return model


if __name__ == '__main__':
    print(get_model_1().summary())
