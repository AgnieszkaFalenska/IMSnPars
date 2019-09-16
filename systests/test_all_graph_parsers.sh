#!/bin/bash

set -ue

## options
D=$(readlink -f $(dirname $0))
source $D/../scripts/get_global_vars.sh

TRAIN=`readlink -ev $D/test_data/train_small.conllu`
DEV=`readlink -ev $D/test_data/dev_small.conllu`

OUT=$D/out-graph
mkdir -p $OUT

DEVICE="CPU"
MODEL=$OUT/test_graph.model
PARSER=$IMSNPARS/imsnpars/main.py

#IFS=',' read -a msts <<< "Eisner,CLE"
#IFS=',' read -a labelers <<< "None,graph-mtl,graph"
#IFS=',' read -a contexts <<< "bilstm,concat"
#IFS='|' read -a feats <<< "h,d,h+1,d-1,dist|h,d|h,d,dist|h+1,d|h,d+1"
#IFS='|' read -a representations <<< "word,char|word,pos|word,morph"
#IFS=',' read -a dummys <<< "zero,random,random_sep"
#
#for mst in "${msts[@]}"; do
#	for dummy in "${dummys[@]}"; do
#	for labeler in "${labelers[@]}"; do
#	for context in "${contexts[@]}"; do
#	for feat in "${feats[@]}"; do
#	for repr in "${representations[@]}"; do
#		echo "dummy="$dummy, "mst="$mst, "feat="$feat, "labeler="$labeler, "context="$context, "repr="$repr
#
		# training
#		echo "Training"
#		$PYTHON $PARSER \
#		        --dynet-devices $DEVICE \
#		        --dynet-seed $DYNET_SEED \
#		        --parser GRAPH \
#		        --train $TRAIN \
#		    	--save $MODEL \
#		    	--dev $DEV \
#		    	--format conllu \
#		    	--epochs 2 \
#		    	--test $DEV \
#		        --representations $repr \
#		        --contextRepr $context \
#				--labeler $labeler \
#				--mst $mst \
#				--features $feat \
#				--dummy $dummy \
#		        --output $MODEL.train.out > $MODEL.train.log
#    	
#		echo "Predicting"
#		$PYTHON $PARSER \
#		        --dynet-devices "CPU" \
#		        --model $MODEL \
#		        --parser GRAPH \
#		        --format conllu \
#		    	--test $DEV \
#		    	--output $MODEL.predict.out > $MODEL.predict.log
#    	
#		echo "Comparing"
#		diff $MODEL.train.out $MODEL.predict.out > $MODEL.diff || echo "DIFFERENT!"
#		echo "OK"
#		
#		rm $MODEL.train.out
#		rm $MODEL.predict.out
#		rm $MODEL
#	done
#	done
#	done
#	done
#	done
#done


#IFS=',' read -a msts <<< "EisnerO2sib"
#IFS=',' read -a labelers <<< "graph-mtl,None"
#IFS=',' read -a contexts <<< "bilstm,concat"
#IFS='|' read -a feats <<< "h,d,s,h+1,d-1,dist|s,h,d|s,h,d,dist|s,h+1,d|s,h,d+1|h,d"
#IFS='|' read -a representations <<< "word,char|word,pos|word,morph"
#IFS=',' read -a dummys <<< "zero,random,random_sep"
#
#for mst in "${msts[@]}"; do
#	for dummy in "${dummys[@]}"; do
#	for labeler in "${labelers[@]}"; do
#	for context in "${contexts[@]}"; do
#	for feat in "${feats[@]}"; do
#	for repr in "${representations[@]}"; do
#		echo "dummy="$dummy, "mst="$mst, "feat="$feat, "labeler="$labeler, "context="$context, "repr="$repr
#
#		echo "Training"
#		$PYTHON $PARSER \
#		        --dynet-devices $DEVICE \
#		        --dynet-seed $DYNET_SEED \
#		        --parser GRAPH \
#		        --train $TRAIN \
#		    	--save $MODEL \
#		    	--dev $DEV \
#		    	--format conllu \
#		    	--epochs 2 \
#		    	--test $DEV \
#		        --representations $repr \
#		        --contextRepr $context \
#				--labeler $labeler \
#				--mst $mst \
#				--features $feat \
#				--dummy $dummy \
#		        --output $MODEL.train.out > $MODEL.train.log
#    	
#		echo "Predicting"
#		$PYTHON $PARSER \
#		        --dynet-devices "CPU" \
#		        --model $MODEL \
#		        --parser GRAPH \
#		        --format conllu \
#		    	--test $DEV \
#		    	--output $MODEL.predict.out > $MODEL.predict.log
#    	
#		echo "Comparing"
#		diff $MODEL.train.out $MODEL.predict.out > $MODEL.diff || echo "DIFFERENT!"
#		echo "OK"
#		
#		rm $MODEL.train.out
#		rm $MODEL.predict.out
#		rm $MODEL
#	done
#	done
#	done
#	done
#	done
#done

IFS=',' read -a msts <<< "EisnerO2g"
IFS=',' read -a labelers <<< "graph-mtl,None"
IFS=',' read -a contexts <<< "bilstm,concat"
IFS='|' read -a feats <<< "h,d|h,d,g,h+1,d-1,dist|g,h,d|g,h,d,dist|g,h+1,d|g,h,d+1"
IFS='|' read -a representations <<< "word,char|word,pos|word,morph"
IFS=',' read -a dummys <<< "zero,random,random_sep"

for mst in "${msts[@]}"; do
	for dummy in "${dummys[@]}"; do
	for labeler in "${labelers[@]}"; do
	for context in "${contexts[@]}"; do
	for feat in "${feats[@]}"; do
	for repr in "${representations[@]}"; do
		echo "dummy="$dummy, "mst="$mst, "feat="$feat, "labeler="$labeler, "context="$context, "repr="$repr

		echo "Training"
		$PYTHON $PARSER \
		        --dynet-devices $DEVICE \
		        --dynet-seed $DYNET_SEED \
		        --parser GRAPH \
		        --train $TRAIN \
		    	--save $MODEL \
		    	--dev $DEV \
		    	--format conllu \
		    	--epochs 2 \
		    	--test $DEV \
		        --representations $repr \
		        --contextRepr $context \
				--labeler $labeler \
				--mst $mst \
				--features $feat \
				--dummy $dummy \
		        --output $MODEL.train.out > $MODEL.train.log
    	
		echo "Predicting"
		$PYTHON $PARSER \
		        --dynet-devices "CPU" \
		        --model $MODEL \
		        --parser GRAPH \
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