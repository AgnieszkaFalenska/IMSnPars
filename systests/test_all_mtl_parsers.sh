#!/bin/bash

set -ue

## options
D=$(readlink -f $(dirname $0))
source $D/../scripts/get_global_vars.sh

TRAIN=`readlink -ev $D/test_data/train_small.conllu`
DEV=`readlink -ev $D/test_data/dev_small.conllu`

OUT=$D/out-mtl
mkdir -p $OUT

DEVICE="CPU"
MODEL=$OUT/test_mtl.model
PARSER=$IMSNPARS/imsnpars/main.py

tlabelers="graph-mtl,trans,None"
glabelers="graph-mtl,graph,None"

IFS=',' read -a tasks <<< "GRAPH,TRANS" 

for task1 in "${tasks[@]}"; do
	for task2 in "${tasks[@]}"; do
	
		if [  $task1 == "GRAPH" ]; then
			labeler1=$glabelers
		else
			labeler1=$tlabelers
		fi
		
		if [  $task2 == "GRAPH" ]; then
			labeler2=$glabelers
		else
			labeler2=$tlabelers
		fi
		
		IFS=',' read -a labelers1 <<< $labeler1
		IFS=',' read -a labelers2 <<< $labeler2
		

		for l1 in "${labelers1[@]}"; do
		for l2 in "${labelers2[@]}"; do
	
			echo "task1="$task1", task2="$task2", labeler1="$l1, "labeler1="$l2

			echo "Training"
			$PYTHON $PARSER \
			        --dynet-devices $DEVICE \
			        --dynet-seed $DYNET_SEED \
			        --parser MTL \
			        --train $TRAIN \
			    	--save $MODEL \
			    	--dev $DEV \
			    	--format conllu \
			    	--epochs 2 \
			    	--test $DEV \
			        --firstTask $task1 \
        			--secondTask $task2 \
        			--t1Labeler $l1 \
        			--t2Labeler $l2 \
			        --output $MODEL.train.out > $MODEL.train.log
	    
			echo "Predicting T1"
			$PYTHON $PARSER \
			        --dynet-devices "CPU" \
			        --dynet-seed $DYNET_SEED \
			        --seed $PARS_SEED \
			        --model $MODEL \
			        --parser MTL \
			        --firstTask $task1 \
			        --secondTask $task2 \
			        --format conllu \
			    	--test $DEV \
			    	--output $MODEL.predict.out > $MODEL.predict_1.log
			
			echo "Predicting T2"
			$PYTHON $PARSER \
			        --dynet-devices "CPU" \
			        --dynet-seed $DYNET_SEED \
			        --seed $PARS_SEED \
			        --model $MODEL \
			        --parser MTL \
			        --firstTask $task1 \
			        --secondTask $task2 \
			        --format conllu \
			    	--test $DEV > $MODEL.predict_2.log
	    	
    	
			echo "Comparing outputs of the first task"
			diff $MODEL.train.out $MODEL.predict.out
			echo "OK"

			echo "Evaluating outputs of the second task"
			IFS=',' read -a TLAS <<< `cat $MODEL.train.log | grep "T2 UAS" | sed -E 's/.*LAS=([^\s]*)/\1/' | tr '\r\n' ','`
			PLAS=`cat $MODEL.predict_2.log | grep "T2 UAS" | sed -E 's/.*LAS=([^\s]*)/\1/'`

			echo "Training:" ${TLAS[-1]}, "prediction: "$PLAS
			
			rm $MODEL*
			
		done
		done
	done
done