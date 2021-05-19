#!/bin/bash

ORG_DATA_DIR="/data/daten/python/master/style-dataset-preprocessing/data/"


DATA_SET="gyafc"
LABEL_0="formal"
LABEL_1="informal"

python3 majority.py \
    --org_data_dir "${ORG_DATA_DIR}${DATA_SET}" \
    --data_set "$DATA_SET" \
    --label_0 "$LABEL_0" \
    --label_1 "$LABEL_1"

token_level="chars"
for n_gram in 1 2 3 4; do
    python3 regression.py \
        --org_data_dir "${ORG_DATA_DIR}${DATA_SET}" \
        --data_set "$DATA_SET" \
        --label_0 "$LABEL_0" \
        --label_1 "$LABEL_1" \
        --n_gram $n_gram \
        --token_level "$token_level"
done

token_level="words"
for n_gram in 1 2; do
    python3 regression.py \
        --org_data_dir "${ORG_DATA_DIR}${DATA_SET}" \
        --data_set "$DATA_SET" \
        --label_0 "$LABEL_0" \
        --label_1 "$LABEL_1" \
        --n_gram $n_gram \
        --token_level "$token_level"
done


DATA_SET="olid"
LABEL_0="NOT"
LABEL_1="OFF"

CLASS_DATA_DIR="${ORG_DATA_DIR}${DATA_SET}/classification"
python3 majority.py \
    --org_data_dir "$CLASS_DATA_DIR" \
    --data_set "$DATA_SET" \
    --label_0 "$LABEL_0" \
    --label_1 "$LABEL_1"

token_level="chars"
for n_gram in 1 2 3 4; do
    python3 regression.py \
        --org_data_dir "$CLASS_DATA_DIR" \
        --data_set "$DATA_SET" \
        --label_0 "$LABEL_0" \
        --label_1 "$LABEL_1" \
        --n_gram $n_gram \
        --token_level "$token_level"
done

token_level="words"
for n_gram in 1 2; do
    python3 regression.py \
        --org_data_dir "$CLASS_DATA_DIR" \
        --data_set "$DATA_SET" \
        --label_0 "$LABEL_0" \
        --label_1 "$LABEL_1" \
        --n_gram $n_gram \
        --token_level "$token_level"
done


DATA_SET="politeness"
LABEL_0="polite"
LABEL_1="impolite"

python3 majority.py \
    --org_data_dir "${ORG_DATA_DIR}${DATA_SET}" \
    --data_set "$DATA_SET" \
    --label_0 "$LABEL_0" \
    --label_1 "$LABEL_1"

token_level="chars"
for n_gram in 1 2 3 4; do
    python3 regression.py \
        --org_data_dir "${ORG_DATA_DIR}${DATA_SET}" \
        --data_set "$DATA_SET" \
        --label_0 "$LABEL_0" \
        --label_1 "$LABEL_1" \
        --n_gram $n_gram \
        --token_level "$token_level"
done

token_level="words"
for n_gram in 1 2; do
    python3 regression.py \
        --org_data_dir "${ORG_DATA_DIR}${DATA_SET}" \
        --data_set "$DATA_SET" \
        --label_0 "$LABEL_0" \
        --label_1 "$LABEL_1" \
        --n_gram $n_gram \
        --token_level "$token_level"
done


DATA_SET="receptiveness"
LABEL_0="receptive"
LABEL_1="not_receptive"
CLASS_DATA_DIR="${ORG_DATA_DIR}${DATA_SET}/classification"

python3 majority.py \
    --org_data_dir "$CLASS_DATA_DIR" \
    --data_set "$DATA_SET" \
    --label_0 "$LABEL_0" \
    --label_1 "$LABEL_1"

token_level="chars"
for n_gram in 1 2 3 4; do
    python3 regression.py \
        --org_data_dir "$CLASS_DATA_DIR" \
        --data_set "$DATA_SET" \
        --label_0 "$LABEL_0" \
        --label_1 "$LABEL_1" \
        --n_gram $n_gram \
        --token_level "$token_level"
done

token_level="words"
for n_gram in 1 2; do
    python3 regression.py \
        --org_data_dir "$CLASS_DATA_DIR" \
        --data_set "$DATA_SET" \
        --label_0 "$LABEL_0" \
        --label_1 "$LABEL_1" \
        --n_gram $n_gram \
        --token_level "$token_level"
done

