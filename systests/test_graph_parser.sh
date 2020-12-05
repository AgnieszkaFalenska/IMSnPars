#!/bin/bash

# options
D=$(dirname $0)
source "${D}/get_global_vars.sh"

TRAIN="${D}/test_data/train_small.conllu"
DEV="${D}/test_data/dev_small.conllu"

OUT="${D}/out-graph"
mkdir -p "${OUT}"

DEVICE="CPU"
MODEL="${OUT}/test_graph.model"
PARSER="${IMSNPARS}/imsnpars/main.py"

echo "Training"
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
		--contextRepr bilstm \
        --representations word,pos \
        --labeler graph-mtl \
        --output $MODEL.train.out | tee $MODEL.train.log
    	
echo "Predicting"
$PYTHON $PARSER \
        --dynet-devices "CPU" \
        --dynet-seed 42 \
        --model $MODEL \
        --parser GRAPH \
        --format conllu \
        --test $DEV \
    	--output $MODEL.predict.out | tee $MODEL.predict.log
    	
echo "Comparing"
diff $MODEL.train.out $MODEL.predict.out > $MODEL.train.diff
echo "OK"

