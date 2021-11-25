# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/22/21
"""
import tensorflow as tf
from tensorflow.keras import layers, initializers

from .ult import stn

STN_SEED = 777


def get_stn_a_model_1(size=28):
    input_layers = layers.Input((size, size, 1))
    x = stn(input_layers)

    x = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.SpatialDropout2D(0.5)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    output_layers = layers.Dense(26, activation="softmax")(x)

    model = tf.keras.Model(input_layers, output_layers)

    return model


def get_stn_a_model_2(size=28):
    input_layers = layers.Input((size, size, 1))
    x = stn(input_layers)

    x = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.SpatialDropout2D(0.5)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    output_layers = layers.Dense(26, activation="softmax")(x)

    model = tf.keras.Model(input_layers, output_layers)

    return model


def get_stn_a_model_3(size=28):
    input_layers = layers.Input((size, size, 1))
    x = stn(input_layers)

    x = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.SpatialDropout2D(0.5)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    output_layers = layers.Dense(26, activation="softmax")(x)

    model = tf.keras.Model(input_layers, output_layers)

    return model


def get_stn_a_model_4(size=28):
    input_layers = layers.Input((size, size, 1))
    x = stn(input_layers)

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
    x = layers.Dropout(0.5)(x)

    output_layers = layers.Dense(26, activation="softmax")(x)

    model = tf.keras.Model(input_layers, output_layers)

    return model


def get_stn_a_model_5(size=28):
    input_layers = layers.Input((size, size, 1))
    x = stn(input_layers)

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
    x = layers.Dropout(0.6)(x)

    output_layers = layers.Dense(26, activation="softmax")(x)

    model = tf.keras.Model(input_layers, output_layers)

    return model


def get_stn_a_model_6(size=28):
    input_layers = layers.Input((size, size, 1))
    x = stn(input_layers)

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

    model = tf.keras.Model(input_layers, output_layers)

    return model


def get_stn_a_model_7(size=28):
    input_layers = layers.Input((size, size, 1))
    x = stn(input_layers)

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
    x = layers.Dropout(0.6)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    output_layers = layers.Dense(26, activation="softmax")(x)

    model = tf.keras.Model(input_layers, output_layers)

    return model


def get_stn_a_model_8(size=28):
    input_layers = layers.Input((size, size, 1))
    x = stn(input_layers)

    x = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)

    output_layers = layers.Dense(26, activation="softmax")(x)

    model = tf.keras.Model(input_layers, output_layers)

    return model


def get_stn_a_model_9(size=28):
    input_layers = layers.Input((size, size, 1))
    x = stn(input_layers)

    x = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)

    x = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)

    output_layers = layers.Dense(26, activation="softmax")(x)

    model = tf.keras.Model(input_layers, output_layers)

    return model
