#!/bin/bash

set -ue

## options
D=$(readlink -f $(dirname $0))
source $D/../scripts/get_global_vars.sh

TRAIN=`readlink -ev $D/test_data/train.conllu`
DEV=`readlink -ev $D/test_data/dev_small.conllu`

OUT=$D/out-graph
mkdir -p $OUT

MODEL=$OUT/test_graph_analysis.model
DEVICE="CPU"

PARSER=$IMSNPARS/imsnpars/main.py

echo "Training graph-based model"
$PYTHON $PARSER \
        --dynet-devices $DEVICE \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --loglevel DEBUG \
        --parser GRAPH \
        --train $TRAIN \
    	--save $MODEL \
    	--dev $DEV \
    	--format conllu \
    	--epochs 2 \
    	--test $DEV \
		--output $MODEL.train.out > $MODEL.train.log
		
ATOOL=$IMSNPARS/imsnpars/analyze.py
OUT=$OUT/test_graph_analysis.model.surf_analysis

echo "Analysing"
$PYTHON $ATOOL \
        --loglevel DEBUG \
        --parser GRAPH \
        --analyze SURF \
        --model $MODEL \
        --test $DEV \
    	--output $OUT

echo "Does it look reasonable?"
cat $OUT | head -n 10

OUT=$D/out-trans
mkdir -p $OUT

MODEL=$OUT/test_trans_analysis.model

echo "Training transition-based model"
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
		--output $MODEL.train.out > $MODEL.train.log
		
ATOOL=$IMSNPARS/imsnpars/analyze.py
OUT=$OUT/test_trans_analysis.model.surf_analysis

echo "Analysing"
$PYTHON $ATOOL \
        --loglevel DEBUG \
        --parser TRANS \
        --analyze SURF \
        --model $MODEL \
        --test $DEV \
    	--output $OUT
    	
echo "Does it look reasonable?"
cat $OUT | head -n 10
    	