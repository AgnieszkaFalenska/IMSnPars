#!/usr/bin/env python3
import os
from pathlib import Path
import requests


if __name__ == '__main__':
    # TODO: IMSnPars test files
    PATH = f"{str(Path.home())}/imsnpars_data/systests"
    os.makedirs(PATH, exist_ok=True)
    URL = "http://???.de/pfad/test_data.fasttext-v123456789.gz"
    os.system(f"wget -O '{PATH}/test_data.fasttext.gz' '{URL}'")
    URL = "http://???.de/pfad/test_data.elmo.ly-2.hdf5.gz"
    os.system(f"wget -O '{PATH}/test_data.elmo.ly-2.hdf5' '{URL}'")

    # TODO: IMSnPars folder with serialized model
    PATH = f"{str(Path.home())}/imsnpars_data/models"
    os.makedirs(PATH, exist_ok=True)
    URL = "http://???.de/pfad/model-trans-20201130/"
    os.system(f"wget -r --no-parent -O '{PATH}/trans' '{URL}'")
