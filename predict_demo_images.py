# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/25/21
"""
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imutils.paths import list_images
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm

from image_pipeline import pipeline_for_demo, HandDetector, BgRemover, get_img_ndarray

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def predict(norm_hand, model):
    norm_hand = norm_hand / .255
    norm_hand = np.expand_dims(norm_hand, 0)
    pred_cls_idx = np.argmax(model.predict(norm_hand))
    pred_cls = ascii_uppercase[pred_cls_idx]

    return pred_cls


def get_norm_hand_ls(hand_img_dir: str) -> list:
    img_path_list = sorted(list(list_images(hand_img_dir)))
    # img_path_list = img_path_list[:1]

    norm_hand_ls = list()
    for image_path in tqdm(img_path_list, total=len(img_path_list)):
        alphabet = image_path.split('/')[-1].split('_')[-1].split('.')[0]
        raw_image = get_img_ndarray(image_path)
        norm_hand = pipeline_for_demo(raw_image, hdt, bgr, img_size=28)
        norm_hand_ls.append((alphabet, raw_image, norm_hand))

        # if norm_hand is not None:
        #     show_img(norm_hand, image_path)

    return norm_hand_ls


def print_prediction(norm_hand_ls: list, model, fig_size=(9, 20)):
    fig = plt.figure(figsize=fig_size)
    columns = 4
    rows = 7

    fc = 0
    ax = []  # ax enables access to manipulate each of subplots
    for i in range(columns * rows):
        if i < 26:
            label = norm_hand_ls[i][0]
            img = norm_hand_ls[i][1]
            norm_hand = norm_hand_ls[i][2]
            pred_cls = predict(norm_hand, model)
            if pred_cls != label:
                fc += 1

            title = f"{label} -> {pred_cls}"
        else:
            img = np.random.randint(1, size=(1, 1))
            title = ''

        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))

        ax[-1].set_title(title)  # set title
        ax[i].set_axis_off()

        plt.imshow(img)

    plt.show()
    acc = fc / 26
    print(f"Accuracy: {acc}")

    return fc, acc


if __name__ == '__main__':
    normal_model_dir = 'saved_model/normal_model'
    normal_model = tf.keras.models.load_model(normal_model_dir)

    model = normal_model

    hdt = HandDetector()
    bgr = BgRemover()
    bgr.load_model()

    image_dir = 'demo-data/jet'
    norm_hand_ls = get_norm_hand_ls(image_dir)
    # print(norm_hand_ls)
    print_prediction(norm_hand_ls, normal_model)

    # norm_hand = norm_hand_ls[0][2]
    # print(norm_hand.shape)
    # pred = predict(norm_hand, model)
    # print(pred)
    # print(norm_hand.shape)

    # a = norm_hand_ls[0]
    # show_img(a[1])
    # show_img(a[2])
