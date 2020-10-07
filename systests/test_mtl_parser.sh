#!/bin/bash

set -ue

## options
D=$(readlink -f $(dirname $0))

source $D/../scripts/get_global_vars.sh

TRAIN=`readlink -ev $D/test_data/train.conllu`
DEV=`readlink -ev $D/test_data/dev.conllu`

OUT=$D/out-mtl
mkdir -p $OUT

DEVICE="CPU"

PARSER=$IMSNPARS/imsnpars/main.py
MODEL=$OUT/test_mtl.model

echo "Training"
$PYTHON $PARSER --parser MTL \
        --dynet-devices $DEVICE \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --train $TRAIN \
    	--save $MODEL \
    	--dev $DEV \
    	--format conllu \
    	--epochs 3 \
    	--test $DEV \
    	--representations word,pos \
        --firstTask GRAPH \
        --secondTask TRANS \
        --t1Labeler graph-mtl \
        --t2Labeler graph-mtl \
        --t2Features s0,s1,b0,s2,s0lmc,s0rmc,s1lmc,s1rmc,s2lmc,s2rmc,b0lmc \
        --output $MODEL.train.out | tee $MODEL.train.log

echo "Predicting T1"
$PYTHON $PARSER \
        --dynet-devices "CPU" \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --model $MODEL \
        --parser MTL \
        --firstTask GRAPH \
        --secondTask TRANS \
        --format conllu \
    	--test $DEV \
    	--output $MODEL.predict.out
#    	
echo "Predicting T2"
$PYTHON $PARSER \
        --dynet-devices "CPU" \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --model $MODEL \
        --parser MTL \
        --firstTask GRAPH \
        --secondTask TRANS \
        --format conllu \
    	--test $DEV | tee $MODEL.predict.log
    	
echo "Comparing outputs of the first task"
diff $MODEL.train.out $MODEL.predict.out
echo "OK"

echo "Evaluating outputs of the second task"
IFS=',' read -a TLAS <<< `cat $MODEL.train.log | grep "T2 UAS" | sed -E 's/.*LAS=([^\s]*)/\1/' | tr '\r\n' ','`
PLAS=`cat $MODEL.predict.log | grep "T2 UAS" | sed -E 's/.*LAS=([^\s]*)/\1/'`

echo "Training:" ${TLAS[-1]}, "prediction: "$PLAS






