#!/bin/bash

DATA_SET="receptiveness"
LABEL_0="receptive"
LABEL_1="not_receptive"
ORG_DATA_DIR_BASE="/data/daten/python/master/style-dataset-preprocessing/data"
ORG_DATA_DIR_SUB="$DATA_SET/classification"
ORG_DATA_DIR="$ORG_DATA_DIR_BASE/$ORG_DATA_DIR_SUB"
CHECKPOINT_DIR="results/best/$DATA_SET/best/"
BATCH_SIZE=256

python3 "eval_classifier.py" \
    --org_data_dir "$ORG_DATA_DIR" \
    --data_set "$DATA_SET" \
    --label_0 "$LABEL_0" \
    --label_1 "$LABEL_1" \
    --eval_batch_size $BATCH_SIZE \
    --checkpoint_dir "$CHECKPOINT_DIR"

