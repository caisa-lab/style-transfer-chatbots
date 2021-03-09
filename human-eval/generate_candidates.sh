GEN_SCRIPT="cmd-demo/predict_service.py"
 
for style in formal informal polite impolite receptive not_receptive offensive not_offensive; 
do
    python3 "$GEN_SCRIPT" \
        --model $style \
        --input_file "human-eval/input.csv" \
        --output_file "human-eval/paraphrases.csv"
done
