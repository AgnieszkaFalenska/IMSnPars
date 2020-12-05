#!/bin/bash

# options
D=$(dirname $0)
source "${D}/get_global_vars.sh"

TRAIN="${D}/test_data/train_small.conllu"
DEV="${D}/test_data/dev_small.conllu"
VEC="${HOME}/imsnpars_data/systests/test_data.fasttext.gz"

OUT="${D}/out-fasttext"
mkdir -p "${OUT}"

DEVICE="CPU"
MODEL="${OUT}/test_fasttext.model"
PARSER="${IMSNPARS}/imsnpars/main.py"

echo "Training"
$PYTHON $PARSER --parser TRANS \
        --dynet-devices $DEVICE \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --train $TRAIN \
    	--save $MODEL \
    	--dev $DEV \
    	--format conllu \
    	--epochs 2 \
    	--test $DEV \
    	--representations word,pos,eword \
    	--embFile $VEC \
        --output $MODEL.train.out | tee $MODEL.train.log
    
echo "Predicting"
$PYTHON $PARSER \
        --dynet-devices "CPU" \
        --dynet-seed $DYNET_SEED \
        --seed $PARS_SEED \
        --model $MODEL \
        --parser TRANS \
        --format conllu \
    	--test $DEV \
    	--output $MODEL.predict.out

echo "Comparing outputs of the first task"
diff $MODEL.train.out $MODEL.predict.out
echo "OK"
