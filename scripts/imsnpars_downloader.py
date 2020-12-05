#!/usr/bin/env python3
import os
from pathlib import Path


if __name__ == '__main__':
    # TODO: IMSnPars test files
    PATH = f"{str(Path.home())}/imsnpars_data/systests"
    os.makedirs(PATH, exist_ok=True)
    URL = "https://www2.ims.uni-stuttgart.de/htdocs/data/imsnpars/test_data.fasttext.gz"
    os.system(f"wget -O '{PATH}/test_data.fasttext.gz' '{URL}'")
    URL = "https://www2.ims.uni-stuttgart.de/data/imsnpars/test_data.elmo.ly-2.hdf5"
    os.system(f"wget -O '{PATH}/test_data.elmo.ly-2.hdf5' '{URL}'")

    # TODO: IMSnPars folder with serialized model
    #PATH = f"{str(Path.home())}/imsnpars_data/models"
    #os.makedirs(PATH, exist_ok=True)
    #URL = "http://???.de/pfad/model-trans-20201130/"
    #os.system(f"wget -r --no-parent -O '{PATH}/trans' '{URL}'")
