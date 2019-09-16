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


IFS='|' read -a toDrop <<< "s0|s0L|s1|s2|b0|s1L|s0R|b2"

for drop in "${toDrop[@]}"; do
	echo "Training for" $drop
	
	rm -f $MODEL.train.out
	
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
	    	--dropContext $drop \
			--output $MODEL.train.out > $MODEL.train.log

	echo "Predicting for" $drop
	rm -f $MODEL.predict.out
	
	$PYTHON $PARSER \
	        --dynet-devices "CPU" \
	        --dynet-seed 42 \
	        --model $MODEL \
	        --parser TRANS \
	        --format conllu \
	    	--test $DEV \
	    	--output $MODEL.predict.out > $MODEL.predict.log

	diff $MODEL.train.out $MODEL.predict.out
	echo "OK"
done

IFS='|' read -a toDrop <<< "s0L_|s1R_"

for drop in "${toDrop[@]}"; do
	echo "Training for" $drop
	
	rm -f $MODEL.train.log
	
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
	    	--dropContext $drop > $MODEL.train.log

	TLAS=`cat $MODEL.train.log | grep ", UAS" | sed -E 's/.*LAS=([^\s]*)/\1/'`
	echo "Training dev LAS" $TLAS
	
	rm -f $MODEL.predict.log
	
	echo "Predicting for" $drop
	$PYTHON $PARSER \
	        --dynet-devices "CPU" \
	        --dynet-seed 42 \
	        --model $MODEL \
	        --parser TRANS \
	        --format conllu \
	    	--test $DEV > $MODEL.predict.log

	PLAS=`cat $MODEL.predict.log | grep "UAS" | sed -E 's/.*LAS=([^\s]*)/\1/'`
	echo "Predicted LAS" $PLAS "(numbers might differ a little bit)"
done
