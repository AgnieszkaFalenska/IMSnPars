#!/bin/bash

set -ue

## options
D=$(readlink -f $(dirname $0))

source $D/../scripts/get_global_vars.sh

TRAIN=`readlink -ev $D/test_data/train.conllu`
DEV=`readlink -ev $D/test_data/dev.conllu`
ELMO=`readlink -ev $HOME/imsnpars_data/systests/test_data.elmo.ly-2.hdf5`
#ELMO=`readlink -ev $D/test_data/ELMo-EWT-default.hdf5`

OUT=$D/out-elmo
mkdir -p $OUT

DEVICE="CPU"

PARSER=$IMSNPARS/imsnpars/main.py
MODEL=$OUT/test_elmo.model

echo "Training"
$PYTHON $PARSER --parser TRANS \
        --dynet-devices $DEVICE \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --train $TRAIN \
    	--save $MODEL \
    	--dev $DEV \
    	--format conllu \
    	--epochs 2 \
    	--test $DEV \
    	--representations word,pos,elmo \
    	--elmoFile $ELMO \
        --output $MODEL.train.out | tee $MODEL.train.log
    
echo "Predicting"
$PYTHON $PARSER \
        --dynet-devices "CPU" \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --model $MODEL \
        --parser TRANS \
        --format conllu \
    	--test $DEV \
    	--elmoFile $ELMO \
    	--output $MODEL.predict.out

echo "Comparing outputs of the first task"
diff $MODEL.train.out $MODEL.predict.out
echo "OK"
