#!/bin/bash

D=$(readlink -f $(dirname $0))
export IMSNPARS=$D/../
export CMD=$IMSNPARS/scripts
export PARS_SEED=688732396
export DYNET_SEED=14464227
export PYTHON=python3