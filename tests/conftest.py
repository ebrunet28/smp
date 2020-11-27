import pytest
from smp import data_dir
from smp.data.extract import extract_data
import os


@pytest.fixture(scope="session")
def download_data():
    if len(os.listdir(data_dir / "raw")) == 1:
        extract_data()
