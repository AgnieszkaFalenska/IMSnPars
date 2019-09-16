#!/bin/bash

set -ue

## options
D=$(readlink -f $(dirname $0))
source $D/../scripts/get_global_vars.sh

TRAIN=`readlink -ev $D/test_data/train_small.conllu`
DEV=`readlink -ev $D/test_data/dev_small.conllu`

OUT=$D/out-trans
mkdir -p $OUT

DEVICE="CPU"
MODEL=$OUT/test_trans.model

PARSER=$IMSNPARS/imsnpars/main.py

IFS=',' read -a tSystems <<< "ArcStandard,ArcHybrid,ASSwap,ArcHybridWithSwap"
IFS=',' read -a labelers <<< "graph-mtl,None,trans"
IFS=',' read -a contexts <<< "bilstm,concat"
IFS='|' read -a representations <<< "word,char|word,pos|word,morph"
IFS='|' read -a features <<< "s0,s1|b0,s1rmc|s2,s1rmc_rmc"
IFS=',' read -a dummys <<< "zero,random,random_sep"

## OTHER OPTIONS
## save max


for tSystem in "${tSystems[@]}"; do
	if [ $tSystem == 'ArcHybrid' ]; then
		IFS=',' read -a oracles <<< "dynamic,static"
	elif [ $tSystem == 'ASSwap' ]; then
		IFS=',' read -a oracles <<< "lazy,eager,static"
	else
		IFS=',' read -a oracles <<< "static"
	fi
	
	for dummy in "${dummys[@]}"; do
	for oracle in "${oracles[@]}"; do
	for labeler in "${labelers[@]}"; do
	for context in "${contexts[@]}"; do
	for repr in "${representations[@]}"; do
	for feat in "${features[@]}"; do
		echo "dummy="$dummy, "tsystem="$tSystem, "oracle="$oracle, "labeler="$labeler, "context="$context, "repr="$repr, "feat="$feat

		# training
		echo "Training"
		$PYTHON $PARSER \
		        --dynet-devices $DEVICE \
		        --dynet-seed $DYNET_SEED \
		        --parser TRANS \
		        --seed $PARS_SEED \
		        --loglevel DEBUG \
		        --train $TRAIN \
		    	--save $MODEL \
		    	--dev $DEV \
		    	--format conllu \
		    	--epochs 2 \
		    	--test $DEV \
		        --representations $repr \
		        --features $feat \
		        --contextRepr $context \
				--labeler $labeler \
				--oracle $oracle \
				--system $tSystem \
				--dummy $dummy \
		        --output $MODEL.train.out > $MODEL.train.log
    	
		echo "Predicting"
		$PYTHON $PARSER \
		        --dynet-devices "CPU" \
		        --model $MODEL \
		        --parser TRANS \
		        --format conllu \
		    	--test $DEV \
		    	--output $MODEL.predict.out > $MODEL.predict.log
    	
		echo "Comparing"
		diff $MODEL.train.out $MODEL.predict.out > $MODEL.diff || echo "DIFFERENT!"
		echo "OK"
		
		rm $MODEL.train.out
		rm $MODEL.predict.out
		rm $MODEL
	done
	done
	done
	done
	done
	done
done