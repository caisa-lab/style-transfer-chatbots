# path to the receptiveness model
STYLE=../man-generation/models/receptive
# decrease the batch size if you get CUDA out of memory errors
BATCH_SIZE=16
# folder where the input file is
DATA_DIR=some/folder
# column name of the text that should be paraphrased
TEXT_COLUMN="text"

python3 predict_service.py \
    --model "$STYLE" \
    --input_file "${DATA_DIR}/input.csv" \
    --text_column "$TEXT_COLUMN" \
    --output_file "${DATA_DIR}/paraphrases.csv" \
    --num_of_candidates 5 \
    --use_cached_intermediate \
    --batch_size $BATCH_SIZE