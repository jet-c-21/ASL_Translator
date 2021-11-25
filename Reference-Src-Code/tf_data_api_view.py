# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/22/21
"""
from cv2 import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import os.path
from pprint import pprint as pp

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

TRAIN_DIR = 'DATASET_A_AP/train'
TEST_DIR = 'DATASET_A_AP/test'
columns = [
    'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X',
    'Y', 'Z'
]


# Convert folder to dataframe of images' paths & labels
def get_paths_labels(path, allowed_extension="jpg"):
    global Path
    images_dir = Path(path)

    file_paths = pd.Series((images_dir.glob(fr'**/*.{allowed_extension}'))).astype(str)
    file_paths.name = "path"

    labels = file_paths.str.split("/")[:].str[-2]
    labels.name = "label"

    # Concatenate file_paths and labels
    df = pd.concat([file_paths, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop=True)
    return df


train_df = get_paths_labels(TRAIN_DIR)
test_df = get_paths_labels(TEST_DIR)

data_generator = ImageDataGenerator(validation_split=0.2, rescale=1. / 255.)
test_generator = ImageDataGenerator(rescale=1. / 255.)

train_images = data_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    subset='training'
)

val_images = data_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    subset='validation'
)

if __name__ == '__main__':
    # pp(dir(train_images))
    print(train_images.image_shape)