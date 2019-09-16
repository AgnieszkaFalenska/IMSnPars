#!/bin/bash

set -ue

## options
D=$(readlink -f $(dirname $0))
source $D/../scripts/get_global_vars.sh

TRAIN=`readlink -ev $D/test_data/train.conllu`
DEV=`readlink -ev $D/test_data/dev.conllu`

OUT=$D/out-graph
mkdir -p $OUT

DEVICE="CPU"

PARSER=$IMSNPARS/imsnpars/main.py

echo "Training"
MODEL=$OUT/test_graph.model


IFS='|' read -a toDrop <<< "h|d"

for drop in "${toDrop[@]}"; do
	echo "Training for" $drop
	
	rm -f $MODEL.train.out
	
	$PYTHON -u $PARSER \
	        --dynet-devices $DEVICE \
	        --dynet-seed $DYNET_SEED \
	        --seed $PARS_SEED \
	        --parser GRAPH \
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
	        --parser GRAPH \
	        --format conllu \
	    	--test $DEV \
	    	--output $MODEL.predict.out > $MODEL.predict.log

	diff $MODEL.train.out $MODEL.predict.out
	echo "OK"
done


$PYTHON -u $PARSER \
	        --dynet-devices $DEVICE \
	        --dynet-seed $DYNET_SEED \
	        --seed $PARS_SEED \
	        --parser GRAPH \
	        --train $TRAIN \
	    	--save $MODEL \
	    	--dev $DEV \
	    	--format conllu \
	    	--epochs 2 \
	    	--test $DEV > $MODEL.train.log

TLAS=`cat $MODEL.train.log | grep ", UAS" | sed -E 's/.*LAS=([^\s]*)/\1/'`
echo "Full model training dev LAS (for iterations)" $TLAS
	
IFS='|' read -a toDrop <<< "h+-1_|d+-1_|h+-2_|s|c"

for drop in "${toDrop[@]}"; do
	echo "Training for" $drop
	
	rm -f $MODEL.train.log

	$PYTHON -u $PARSER \
	        --dynet-devices $DEVICE \
	        --dynet-seed $DYNET_SEED \
	        --seed $PARS_SEED \
	        --parser GRAPH \
	        --train $TRAIN \
	    	--save $MODEL \
	    	--dev $DEV \
	    	--format conllu \
	    	--epochs 2 \
	    	--test $DEV \
	    	--dropContext $drop > $MODEL.train.log

	TLAS=`cat $MODEL.train.log | grep ", UAS" | sed -E 's/.*LAS=([^\s]*)/\1/'`
	echo "Training dev LAS (for iterations)" $TLAS
	
	rm -f $MODEL.predict.log
	
	echo "Predicting for" $drop
	$PYTHON $PARSER \
	        --dynet-devices "CPU" \
	        --dynet-seed 42 \
	        --model $MODEL \
	        --parser GRAPH \
	        --format conllu \
	    	--test $DEV > $MODEL.predict.log

	PLAS=`cat $MODEL.predict.log | grep "UAS" | sed -E 's/.*LAS=([^\s]*)/\1/'`
	echo "Predicted LAS" $PLAS "(numbers might differ a little bit)"
done

