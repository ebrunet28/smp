import kaggle
from smp import data_dir
import zipfile
import os


def extract_data():
    """
    1. Download data from kaggle
    2. Upzip file
    3. Remove zip file

    :return:
    """
    kaggle.api.competition_download_files("ift6758-a20", path=data_dir/"raw")

    zip_file_name = data_dir/"raw"/"ift6758-a20.zip"

    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(data_dir/"raw")

    os.remove(zip_file_name)


if __name__ == "__main__":
    extract_data()





