# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/20/21
"""
import zipfile
import gdown
from tqdm import tqdm


def extract_zip(zip_path: str, extract_path=None):
    print('Extracting zip...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            if extract_path:
                zip_ref.extract(member=file, path=extract_path)
            else:
                zip_ref.extract(member=file, path='.')
    print('Finish extracting')


def download_gd_zip(file_id: str) -> str:
    d_url = f"https://drive.google.com/uc?id={file_id}"
    zip_path = f"{file_id}.zip"
    gdown.download(d_url, zip_path, quiet=False)
    return zip_path


def download_file_by_gdown(file_id: str, save_fp=None):
    zip_path = download_gd_zip(file_id)
    extract_zip(zip_path, save_fp)


if __name__ == '__main__':
    dataset_a_file_id = '1q1IjtWrWjWfJSUG55s1SN0w3bOTuG7cV'
    dataset_a_ap_file_id = '1TKh-0eYwWLF41tfCfykCuvBVpSY7Fj4k'

    file_id_ls = [
        dataset_a_file_id,
        dataset_a_ap_file_id,
    ]

    for i, file_id in enumerate(file_id_ls, start=1):
        download_file_by_gdown(file_id)
