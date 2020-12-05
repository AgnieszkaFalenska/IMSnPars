#!/usr/bin/env python3
import os
from pathlib import Path
import argparse


if __name__ == '__main__':
    # parse script's input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--systests', action='store_true',
        help='Download large files for system tests.')
    parser.add_argument(
        '--hdt', action='store_true',
        help='Download Hamburg Treebank training set.')
    args = parser.parse_args()

    # IMSnPars test files
    if args.systests:
        PATH = f"{str(Path.home())}/imsnpars_data/systests"
        os.makedirs(PATH, exist_ok=True)
        URL = "https://www2.ims.uni-stuttgart.de/htdocs/data/imsnpars/test_data.fasttext.gz"
        os.system(f"wget -O '{PATH}/test_data.fasttext.gz' '{URL}'")
        URL = "https://www2.ims.uni-stuttgart.de/data/imsnpars/test_data.elmo.ly-2.hdf5"
        os.system(f"wget -O '{PATH}/test_data.elmo.ly-2.hdf5' '{URL}'")

    # Demo training set: HDT
    if args.hdt:
        PATH = f"{str(Path.home())}/imsnpars_data/hdt"
        TMP = f"{str(Path.home())}/imsnpars_data/tmp"
        os.makedirs(PATH, exist_ok=True)
        os.makedirs(TMP, exist_ok=True)
        URL = "https://github.com/UniversalDependencies/UD_German-HDT/archive/r2.7.tar.gz"
        os.system(f"wget -O '{TMP}/r2.7.tar.gz' '{URL}'")
        os.system(f"tar -C '{TMP}' -xvf '{TMP}/r2.7.tar.gz'")
        os.system(f"cat $(find {TMP} -name 'de_hdt-ud-train-*.conllu') > {PATH}/train.conllu")
        os.system(f"cat $(find {TMP} -name 'de_hdt-ud-dev.conllu') > {PATH}/dev.conllu")
        os.system(f"cat $(find {TMP} -name 'de_hdt-ud-test.conllu') > {PATH}/test.conllu")
        os.system(f"rm -r {TMP}")

    # TODO: IMSnPars folder with serialized model
    #PATH = f"{str(Path.home())}/imsnpars_data/models"
    #os.makedirs(PATH, exist_ok=True)
    #URL = "http://???.de/pfad/model-trans-20201130/"
    #os.system(f"wget -r --no-parent -O '{PATH}/trans' '{URL}'")
