# path to the receptiveness model
STYLE=receptive
# decrease the batch size if you get CUDA out of memory errors
BATCH_SIZE=16
# folder where the input file is
DATA_DIR=.
# column name of the text that should be paraphrased
TEXT_COLUMN="text"
# top_p sampling value for the intermediate paraphraser
TOP_P_PARAPHRASE=0.12
# top_p sampling value for the inverse paraphraser (performs the actual style transfer)
TOP_P_STYLE=0.34

python3 predict_service.py \
    --model "$STYLE" \
    --input_file "${DATA_DIR}/input.csv" \
    --text_column "$TEXT_COLUMN" \
    --output_file "${DATA_DIR}/paraphrases.csv" \
    --num_of_candidates 5 \
    --use_cached_intermediate \
    --batch_size $BATCH_SIZE \
    --top_p_paraphrase $TOP_P_PARAPHRASE \
    --top_p_style $TOP_P_STYLE