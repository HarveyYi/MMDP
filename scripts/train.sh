#!/bin/bash

# custom config
CUDA_VISIBLE_DEVICES=$1
TIME=$2
INDEX=$3
DATASETS=("gbmlgg")

MODEL=UMEML
TASK=Survival
DATA=datasets
DATASET=${DATASETS[$INDEX]}
for DATASET in ${DATASETS[@]}
do
    for FOLD in 0 1 2 3 4 
    do
        DIR=COMPARE/exp${TIME}/train/${TASK}/${MODEL}/${DATASET}/fold${FOLD}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Resuming..."
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py \
            --root ${DATA} \
            --dataname ${DATASET} \
            --fold ${FOLD} \
            --trainer MMTRAIN \
            --config-file configs/${TASK}/${MODEL}.yaml \
            --output-dir ${DIR} 

        else
            echo "Run this job and save the output to ${DIR}"
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tools/train.py \
            --root ${DATA} \
            --dataname ${DATASET} \
            --fold ${FOLD} \
            --trainer MMTRAIN \
            --config-file configs/${TASK}/${MODEL}.yaml \
            --output-dir ${DIR} 
        fi
    done
done

