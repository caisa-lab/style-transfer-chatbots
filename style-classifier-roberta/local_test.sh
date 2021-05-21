#!/bin/sh
DATA_DIR="/data/daten/python/master/style-dataset-preprocessing/data/gyafc"

# replace gyafc with name of dataset!
# also replace labels!

batchSize=16
lr=1e-4
MAX_EPOCHS=10

python3 train_classifier.py \
    --org_data_dir "$DATA_DIR" \
    --data_set "gyafc" \
    --label_0 "formal" \
    --label_1 "informal" \
    --eval_batch_size 512 \
    --freeze_last_layers 0 \
    --train_batch_size $batchSize \
    --learning_rate $lr \
    --max_epochs $MAX_EPOCHS

