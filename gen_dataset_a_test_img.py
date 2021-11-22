# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/20/21
"""
from collections import Counter

# from tqdm import tqdm
import datetime
from cv2 import cv2
from imutils.paths import list_images
import multiprocessing as mp
import random
import datetime

from data_tool import *
from image_pipeline import *
from image_pipeline.preprocessing import hash_image

SEED = 777
random.seed(SEED)


def get_norm_hand(image_path: str, hdt: HandDetector, bgr: BgRemover) -> Union[np.ndarray, None]:
    raw_image = get_img_ndarray(image_path)
    if raw_image is None:
        return

    norm_hand = pipeline_base(raw_image, hdt, bgr)
    if norm_hand is None:
        return

    return norm_hand


def check_alphabet_complete(alphabet_test_img_dir, alphabet_test_img_ap_dir) -> bool:
    a = os.listdir(alphabet_test_img_dir)
    b = os.listdir(alphabet_test_img_ap_dir)

    print(f"{alphabet_test_img_dir} : {len(a)}  {alphabet_test_img_ap_dir} : {len(b)}")

    if len(a) >= TEST_IMG_TARGET and len(b) >= TEST_IMG_TARGET:
        return True
    else:
        return False


def _get_img_hash(img_path) -> Union[str, None]:
    # print(img_path)
    img = get_img_ndarray(img_path)
    if img is not None:
        return hash_image(img)
    else:
        print('failed to hash image')


def get_img_hash_mp(img_path_ls):
    print('getting image hash...')
    with mp.Pool() as pool:
        result = list(
            tqdm(
                pool.imap(_get_img_hash, img_path_ls), total=len(img_path_ls)
            )
        )

    return result


def adding_alphabet_test_image(alphabet: str, img_path_ls: list, img_hash_set: set, save_dir: str, save_ap_dir: str):
    # print(img_path_ls)
    # print(img_hash_set)
    # print(save_dir)
    # print(save_ap_dir)
    for img_path in tqdm(img_path_ls, total=len(img_path_ls)):
        raw_image = get_img_ndarray(img_path)

        if raw_image is None:
            continue

        img_hash = hash_image(raw_image)
        if img_hash in img_hash_set:
            print(f"counter identical image: {img_path}")
            continue

        norm_hand = get_norm_hand(img_path, hdt, bgr)
        if norm_hand is None:
            continue

        img_name = get_img_save_name(alphabet, is_train=False)
        raw_img_s_path = f"{save_dir}/{img_name}.jpg"
        norm_hand_s_path = f"{save_ap_dir}/{img_name}.jpg"
        # print(raw_img_s_path, norm_hand_s_path)

        cv2.imwrite(raw_img_s_path, raw_image)
        # show_img(raw_image)
        cv2.imwrite(norm_hand_s_path, norm_hand)
        # show_img(norm_hand)
        print('new image added!')

        img_hash_set.add(img_hash)

        if check_alphabet_complete(save_dir, save_ap_dir):
            return


def alphabet_test_image_generator(alphabet: str, train_alphabet_dir: str):
    img_hash_set = set()
    alphabet_test_img_dir = f"{OUTPUT_DIR_ROOT}/{alphabet}"
    create_dir(alphabet_test_img_dir)
    alphabet_test_img_ap_dir = f"{OUTPUT_AP_DIR_ROOT}/{alphabet}"
    create_dir(alphabet_test_img_ap_dir)

    # get image hash of training image
    # img_hash_from_train = get_img_hash_mp(list(list_images(train_alphabet_dir)))
    # img_hash_set.update(img_hash_from_train)
    # print(f"img hash count: {len(img_hast_set)}")

    # add image from dataset4
    d4_dir = DATASET4_DIR_PATH
    d4_alphabet_dir = f"{d4_dir}/{alphabet}"
    d4_alphabet_img_path_ls = list(list_images(d4_alphabet_dir))
    random.shuffle(d4_alphabet_img_path_ls)
    print(f"d4 candidate for {alphabet}: {len(d4_alphabet_img_path_ls)}")

    adding_alphabet_test_image(alphabet, d4_alphabet_img_path_ls, img_hash_set,
                               alphabet_test_img_dir, alphabet_test_img_ap_dir)
    if check_alphabet_complete(alphabet_test_img_dir, alphabet_test_img_ap_dir):
        return

    # add image from dataset2
    d2_dir = DATASET2_DIR_PATH
    d2_alphabet_dir = f"{d2_dir}/{alphabet}"
    d2_alphabet_img_path_ls = list(list_images(d2_alphabet_dir))
    random.shuffle(d2_alphabet_img_path_ls)
    print(f"d2 candidate for {alphabet}: {len(d2_alphabet_img_path_ls)}")

    adding_alphabet_test_image(alphabet, d2_alphabet_img_path_ls, img_hash_set,
                               alphabet_test_img_dir, alphabet_test_img_ap_dir)
    if check_alphabet_complete(alphabet_test_img_dir, alphabet_test_img_ap_dir):
        return

    # add image from dataset3
    d3_dir = DATASET3_DIR_PATH
    d3_alphabet_dir = f"{d3_dir}/{alphabet}"
    d3_alphabet_img_path_ls = list(list_images(d3_alphabet_dir))
    random.shuffle(d3_alphabet_img_path_ls)
    print(f"d3 candidate for {alphabet}: {len(d3_alphabet_img_path_ls)}")

    adding_alphabet_test_image(alphabet, d3_alphabet_img_path_ls, img_hash_set,
                               alphabet_test_img_dir, alphabet_test_img_ap_dir)
    if check_alphabet_complete(alphabet_test_img_dir, alphabet_test_img_ap_dir):
        return


def get_exe_time_str(exe_time: datetime.timedelta) -> str:
    seconds = exe_time.total_seconds()
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    exe_time_str = f"{hours}hr_{minutes}m_{seconds}s"

    return exe_time_str


def main():
    for train_alphabet_dir in get_alphabet_dir_ls(TRAIN_DIR_ROOT):
        alphabet = train_alphabet_dir.split('/')[-1]
        print(f"generating test image for alphabet: {alphabet}...")
        alphabet_test_image_generator(alphabet, train_alphabet_dir)
        print(f"Finish generating test image for alphabet: {alphabet} \n")


if __name__ == '__main__':
    TEST_IMG_TARGET = 555

    TRAIN_DIR_ROOT = 'TRAIN_DATASET_A'
    TRAIN_AP_DIR_ROOT = 'TRAIN_DATASET_A_AP'

    OUTPUT_DIR_ROOT = 'TEST_DATASET_A'
    OUTPUT_AP_DIR_ROOT = 'TEST_DATASET_A_AP'
    create_dir(OUTPUT_DIR_ROOT)
    create_dir(OUTPUT_AP_DIR_ROOT)

    hdt = HandDetector()

    bgr = BgRemover()
    bgr.load_model()

    DATASET2_DIR_PATH = 'dataset2'
    DATASET3_DIR_PATH = 'dataset3'
    DATASET4_DIR_PATH = 'dataset4'

    s = datetime.datetime.now()
    main()
    e = datetime.datetime.now()
    exe_time = e - s

    exe_time_str = get_exe_time_str(exe_time)
    print(f"time cost = {exe_time_str}")
