# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/22/21
"""
from imutils.paths import list_images
import os


def rename_images(alphabet_dir: str):
    image_path_ls = list(list_images(alphabet_dir))
    image_path_ls.sort()

    for i, image_path in enumerate(image_path_ls, start=1):
        img_name = image_path.split('.jpg')[0].split('/')[-1]

        name_tokens = img_name.split('_')

        old_image_path = image_path
        new_image_path = f"{alphabet_dir}/{name_tokens[0]}_{name_tokens[1]}_{i}.jpg"

        print(old_image_path, new_image_path)

        os.rename(old_image_path, new_image_path)

        # break


def helper(mode: str):
    dir_path = f"{ROOT_DIR_PATH}/{mode}"
    for alphabet in os.listdir(dir_path):
        alphabet_dir = f"{dir_path}/{alphabet}"
        rename_images(alphabet_dir)
        # break


def main():
    helper('train')
    helper('test')


if __name__ == '__main__':
    # ROOT_DIR_PATH = 'DATASET_A'
    ROOT_DIR_PATH = 'DATASET_A_AP'
    main()
