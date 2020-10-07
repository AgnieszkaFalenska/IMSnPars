#!/bin/bash

DIR=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

export IMSNPARS=$DIR/../
export PARS_SEED=688732396
export DYNET_SEED=14464227
export PYTHON=python3
