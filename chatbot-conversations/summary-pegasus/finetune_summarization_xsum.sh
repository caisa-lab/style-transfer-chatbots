#!/bin/sh
#SBATCH --job-name=finetune_pegasus_debate_sum_xsum
#SBATCH --output="/ukp-storage-1/nothvogel/summary-pegasus/logs/log_debate_sum_xsum.txt"
#SBATCH --partition=cai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40GB
#SBATCH -d singleton
#SBATCH --ntasks=1
#SBATCH --mail-user=phil-n@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --account=cai-student

# Experiment Details :- Pegasus model fine-tuning on debate sum corpus
# Run Details :- filtered corpus agressively by compression ratio


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/ukp-storage-1/nothvogel/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/ukp-storage-1/nothvogel/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/ukp-storage-1/nothvogel/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/ukp-storage-1/nothvogel/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

module purge
module load cuda/11.0
conda activate pegasus
export WANDB_API_KEY=yourApiKey
export WANDB_CONFIG_DIR=/ukp-storage-1/nothvogel/.config/wandb
export NLTK_DATA="/ukp-storage-1/nothvogel/.cache/nltk"


MODEL_NAME="google/pegasus-xsum"
DATA_DIR="datasets/debate-sum"
TRAIN_FILE="${DATA_DIR}/train.csv"
VAL_FILE="${DATA_DIR}/val.csv"
TEXT_COLUMN="document"
SUMMARY_COLUMN="abstract"
OUT_DIR="results/debate-sum_xsum"
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=16
MAX_TARGET_LENGTH=64

export TOKENIZERS_PARALLELISM=false

python run_summarization.py \
    --model_name_or_path $MODEL_NAME \
    --do_train \
    --do_eval \
    --task summarization \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --output_dir "$OUT_DIR" \
    --overwrite_output_dir \
    --per_device_train_batch_size=$TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size=$EVAL_BATCH_SIZE \
    --predict_with_generate \
    --text_column "$TEXT_COLUMN" \
    --summary_column "$SUMMARY_COLUMN" \
    --max_source_length 512 \
    --max_target_length $MAX_TARGET_LENGTH \
    --evaluation_strategy epoch \
    --adafactor=True \
    --report_to wandb \
    --num_train_epochs 10 \
    --load_best_model_at_end=True \
    --metric_for_best_model rougeLsum \
    --greater_is_better=True 
    