# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/22/21
"""
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

    model.add(layers.Dense(26, activation='softmax'))

    return model


def get_model_2():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))

    # finish feature extraction
    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))  # 0.25 -> 0.5

    # new dense layer
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(26, activation='softmax'))

    return model


def get_model_3():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())

    # model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    # model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    # model.add(layers.MaxPool2D((2, 2)))

    # finish feature extraction
    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))  # 0.25 -> 0.5

    # new dense layer
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(26, activation='softmax'))

    return model
