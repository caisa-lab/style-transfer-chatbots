GEN_SCRIPT="/data/daten/python/master/style-transfer-paraphrase/cmd-demo/predict_service.py"
DATA_DIR="paraphrases/debate-conversations-1"
BATCH_SIZE=16
 
for style in formal informal polite impolite receptive not_receptive; 
do
    python3 "$GEN_SCRIPT" \
        --model $style \
        --input_file "${DATA_DIR}/input.csv" \
        --output_file "${DATA_DIR}/paraphrases.csv" \
        --num_of_candidates 5 \
        --filter_column "isConst" \
        --filter_false \
        --use_cached_intermediate \
        --batch_size $BATCH_SIZE
done


