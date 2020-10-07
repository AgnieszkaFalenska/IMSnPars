#!/bin/bash

set -ue

## options
D=$(readlink -f $(dirname $0))
source $D/../scripts/get_global_vars.sh

TRAIN=`readlink -ev $D/test_data/train.conllu`
DEV=`readlink -ev $D/test_data/dev.conllu`

OUT=$D/out-stag
mkdir -p $OUT

echo "Converting gold data to stags"
TRAIN_STAG=$OUT/train.gold-stag.conllu
DEV_STAG=$OUT/dev.gold-stag.conllu

python3 $IMSNPARS/imsnpars/stack/stag.py $TRAIN $TRAIN $TRAIN_STAG
python3 $IMSNPARS/imsnpars/stack/stag.py $DEV $DEV $DEV_STAG

MODEL=$OUT/test_graph.gold-stag.model
DEVICE="CPU"

PARSER=$IMSNPARS/imsnpars/main.py

echo "Training graph-based parser with supertags"
$PYTHON $PARSER \
        --dynet-devices $DEVICE \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --loglevel DEBUG \
        --parser GRAPH \
        --train $TRAIN_STAG \
    	--save $MODEL \
    	--dev $DEV_STAG \
    	--format conlluStack \
    	--epochs 2 \
    	--test $DEV_STAG \
		--contextRepr bilstm \
        --representations word,pos \
        --stackRepr stag \
        --labeler "graph-mtl" \
        --output $MODEL.train.out
        
echo "Predicting"
$PYTHON $PARSER \
        --dynet-devices "CPU" \
        --dynet-seed 42 \
        --model $MODEL \
        --parser GRAPH \
        --format conlluStack \
        --test $DEV_STAG \
    	--output $MODEL.predict.out
    	
echo "Comparing"
diff $MODEL.train.out $MODEL.predict.out > $MODEL.train.diff
echo "OK"

echo "Converting predicted data to stags"
DEV_STAG=$OUT/dev.pred-stag.conllu
python3 $IMSNPARS/imsnpars/stack/stag.py $DEV $MODEL.predict.out $DEV_STAG

MODEL=$OUT/test_trans.pred-stag.model

echo "Training transition-based parser with supertags"
$PYTHON $PARSER \
        --dynet-devices $DEVICE \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --loglevel DEBUG \
        --parser TRANS \
        --train $TRAIN_STAG \
    	--save $MODEL \
    	--dev $DEV_STAG \
    	--format conlluStack \
    	--epochs 2 \
    	--test $DEV_STAG \
		--contextRepr bilstm \
        --representations word,pos,stag \
        --labeler "graph-mtl" \
        --output $MODEL.train.out
        
echo "Predicting"
$PYTHON $PARSER \
        --dynet-devices "CPU" \
        --dynet-seed 42 \
        --model $MODEL \
        --parser TRANS \
        --format conlluStack \
        --test $DEV_STAG \
    	--output $MODEL.predict.out
    	
echo "Comparing"
diff $MODEL.train.out $MODEL.predict.out > $MODEL.train.diff
echo "OK"
