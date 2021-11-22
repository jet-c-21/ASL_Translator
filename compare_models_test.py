# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/22/21
"""

from inspect import getmembers, isfunction
from pprint import pprint as pp

import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import asl_model


# Convert folder to DataFrame of images' paths & labels
def get_paths_labels(path, allowed_extension="jpg"):
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


def main():
    ls = list(getmembers(asl_model, isfunction))

    for model_func_name, model_func_obj in reversed(ls):
        if model_func_name in ['stn']:
            continue

        print(f"\nmodel_func_name : {model_func_name}")
        model = model_func_obj()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print('\nTraining...')
        history = model.fit(
            train_images,
            validation_data=val_images,
            epochs=epoch,
        )

        print('\nEvaluating...')
        result = model.evaluate(test_images)
        print(f"{model_func_name} : test loss = {round(result[0], 2)}, test acc= {round(result[1], 2)}")

        acc = result[1]

        record = {
            'acc': acc,
            'history': history,
        }

        model_dict[model_func_name] = record


if __name__ == '__main__':
    TRAIN_DIR = 'DATASET_A_AP/train'
    TEST_DIR = 'DATASET_A_AP/test'

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
        batch_size=128,
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
        batch_size=128,
        shuffle=True,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='label',
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        # batch_size=1,
        shuffle=False,
    )

    epoch = 3
    lr = 0.005
    model_dict = dict()

    # np.random.seed(777)

    main()

    pp(model_dict)
