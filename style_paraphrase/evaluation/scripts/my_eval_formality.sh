#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0



BATCH_SIZE=16
CORPUS="gyafc"
LABEL_0="formal"
STYLE_0="formal"
LABEL_1="informal"
STYLE_1="informal"
ORG_DATA_DIR="/data/daten/python/master/style-dataset-preprocessing/data/$CORPUS"
OUTPUT_DIR="/data/daten/python/master/style-transfer-paraphrase/final-eval/$CORPUS"
CLASS_CHECKPT_DIR="/data/daten/python/master/gyafc-classifier/results/best/${CORPUS}/best"
 
GEN_SCRIPT="/data/daten/python/master/style-transfer-paraphrase/cmd-demo/predict_service.py"
# separate labels / styles
SPLIT_SCRIPT="/data/daten/python/master/style-transfer-paraphrase/style_paraphrase/evaluation/scripts/separate-labels.py"
# script to extract column from df
EXTRACT_SCRIPT="/data/daten/python/master/style-transfer-paraphrase/style_paraphrase/evaluation/scripts/extract-column.py"
EXTRACT_COL="style_transfer"
# script for style classifier
CLASS_SCRIPT="/data/daten/python/master/gyafc-classifier/style_eval_roberta.py"

mkdir -p "$OUTPUT_DIR"
split="test"

python3 "$SPLIT_SCRIPT" \
    --input_csv ${ORG_DATA_DIR}/test.csv


# from style 0 to style 1
paraphraseCsv="${OUTPUT_DIR}/paraphrases_to_${STYLE_1}.csv"
testCsv="${ORG_DATA_DIR}/test.${LABEL_0}.csv"
python3 "$GEN_SCRIPT" \
    --model $STYLE_1 \
    --input_file "$testCsv" \
    --output_file "$paraphraseCsv" \
    --num_of_candidates 5 \
    --use_cached_intermediate \
    --batch_size $BATCH_SIZE

# extract stylistic paraphrase from csv to txt
paraphraseTxt="${OUTPUT_DIR}/transfer_${STYLE_1}_${split}.txt"
python3 "$EXTRACT_SCRIPT" \
    --input_csv "$paraphraseCsv" \
    --output_txt "$paraphraseTxt" \
    --column_name "$EXTRACT_COL"

# extract original sentence from csv to txt
inputRefTxt="${OUTPUT_DIR}/transfer_${STYLE_1}_${split}_input.txt"
python3 "$EXTRACT_SCRIPT" \
    --input_csv "$testCsv" \
    --output_txt "$inputRefTxt" \
    --column_name "original_sentence"

# classify style acc with roberta
python3 "$CLASS_SCRIPT" \
    --input_txt "$paraphraseTxt" \
    --output_txt "${paraphraseTxt}.roberta_labels" \
    --checkpoint_dir "$CLASS_CHECKPT_DIR" \
    --label_0 "$LABEL_0" \
    --label_1 "$LABEL_1" \
    --expected_label "$LABEL_1"

# classify grammatical acceptability
python3 style_paraphrase/evaluation/scripts/acceptability.py \
    --input_file "$paraphraseTxt"

# get similarity with inputs
printf "\nParaphrase scores --- generated vs inputs..\n\n"
python style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py \
    --generated_path "$paraphraseTxt" \
    --reference_strs reference \
    --reference_paths "$inputRefTxt" \
    --output_path ${OUTPUT_DIR}/generated_vs_inputs.txt

printf "\n final normalized scores vs input..\n\n"
python3 style_paraphrase/evaluation/scripts/micro_eval.py \
    --classifier_file "${paraphraseTxt}.roberta_labels" \
    --paraphrase_file "${parapraseTxt}.pp_scores" \
    --generated_file "$paraphraseTxt" \
    --acceptability_file "${paraphraseTxt}.acceptability_labels"



# from style 1 to style 0
paraphraseCsv="${OUTPUT_DIR}/paraphrases_to_${STYLE_0}.csv"
testCsv="${ORG_DATA_DIR}/test.${LABEL_1}.csv"
python3 "$GEN_SCRIPT" \
    --model $STYLE_0 \
    --input_file "$testCsv" \
    --output_file "$paraphraseCsv" \
    --num_of_candidates 5 \
    --use_cached_intermediate \
    --batch_size $BATCH_SIZE

# extract stylistic paraphrase from csv to txt
paraphraseTxt="${OUTPUT_DIR}/transfer_${STYLE_0}_${split}.txt"
python3 "$EXTRACT_SCRIPT" \
    --input_csv "$paraphraseCsv" \
    --output_txt "$paraphraseTxt" \
    --column_name "$EXTRACT_COL"

# extract original sentence from csv to txt
inputRefTxt="${OUTPUT_DIR}/transfer_${STYLE_0}_${split}_input.txt"
python3 "$EXTRACT_SCRIPT" \
    --input_csv "$testCsv" \
    --output_txt "$inputRefTxt" \
    --column_name "original_sentence"

# classify style acc with roberta
python3 "$CLASS_SCRIPT" \
    --input_txt "$paraphraseTxt" \
    --output_txt "${paraphraseTxt}.roberta_labels" \
    --checkpoint_dir "$CLASS_CHECKPT_DIR" \
    --label_0 "$LABEL_0" \
    --label_1 "$LABEL_1" \
    --expected_label "$LABEL_0"

# classify grammatical acceptability
python3 style_paraphrase/evaluation/scripts/acceptability.py \
    --input_file "$paraphraseTxt"

# get similarity with inputs
printf "\nParaphrase scores --- generated vs inputs..\n\n"
python style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py \
    --generated_path "$paraphraseTxt" \
    --reference_strs reference \
    --reference_paths "$inputRefTxt" \
    --output_path ${OUTPUT_DIR}/generated_vs_inputs.txt

printf "\n final normalized scores vs input..\n\n"
python3 style_paraphrase/evaluation/scripts/micro_eval.py \
    --classifier_file "${paraphraseTxt}.roberta_labels" \
    --paraphrase_file "${parapraseTxt}.pp_scores" \
    --generated_file "$paraphraseTxt" \
    --acceptability_file "${paraphraseTxt}.acceptability_labels"
