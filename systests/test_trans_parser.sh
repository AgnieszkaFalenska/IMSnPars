#!/bin/bash

set -ue

## options
D=$(readlink -f $(dirname $0))
source $D/../scripts/get_global_vars.sh

TRAIN=`readlink -ev $D/test_data/train.conllu`
DEV=`readlink -ev $D/test_data/dev.conllu`

OUT=$D/out-trans
mkdir -p $OUT

DEVICE="CPU"


PARSER=$IMSNPARS/imsnpars/main.py

echo "Training"
MODEL=$OUT/test_trans.model

$PYTHON -u $PARSER \
        --dynet-devices $DEVICE \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --parser TRANS \
        --train $TRAIN \
    	--save $MODEL \
    	--dev $DEV \
    	--format conllu \
    	--epochs 2 \
    	--test $DEV \
        --representations word,pos \
        --contextRepr bilstm \
		--features s0,s1,b0 \
		--labeler trans \
		--output $MODEL.train.out #| tee $MODEL.log

echo "Predicting 1"
$PYTHON $PARSER \
        --dynet-devices "CPU" \
        --dynet-seed 42 \
        --model $MODEL \
        --parser TRANS \
        --format conllu \
    	--test $DEV \
    	--output $MODEL.predict.out
    	
echo "Predicting 2"
$PYTHON $PARSER \
        --dynet-devices "CPU" \
        --dynet-seed 42 \
        --model $MODEL \
        --parser TRANS \
        --format conllu \
    	--test $DEV \
    	--output $MODEL.predict.out.2
#    	
echo "Comparing"
diff $MODEL.train.out $MODEL.predict.out > $MODEL.train.diff
diff $MODEL.predict.out $MODEL.predict.out.2 > $MODEL.predict.diff
echo "OK"
