GEN_SCRIPT="/data/daten/python/master/style-transfer-paraphrase/cmd-demo/predict_service.py"

OUT_FILE="/data/daten/python/master/human-eval/paraphrases/chaining/paraphrases"

style=formal
NEXT_OUT_FILE="$OUT_FILE.$style.csv"
python3 "$GEN_SCRIPT" \
    --model $style \
    --input_file "input.csv" \
    --output_file "$NEXT_OUT_FILE" \
    --filter_candidates=True \
    --no_diverse_paraphraser
 
for style in polite receptive;
do
    PREV_OUT_FILE="$NEXT_OUT_FILE"
    NEXT_OUT_FILE="$OUT_FILE.$style.csv"

    python3 "$GEN_SCRIPT" \
        --model $style \
        --input_file "$PREV_OUT_FILE" \
        --output_file "$NEXT_OUT_FILE" \
        --text_column "style_transfer" \
        --filter_candidates=True \
        --no_diverse_paraphraser
done
