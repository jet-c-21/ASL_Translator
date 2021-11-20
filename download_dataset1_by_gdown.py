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
    print('Extracting zip ...')
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
    dataset1_file_id = '1FpYkbhAb7fX1z_ygNA1i__av2h6uAkUd'
    download_file_by_gdown(dataset1_file_id)
