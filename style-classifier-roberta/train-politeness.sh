#!/bin/bash
#SBATCH --job-name=finetune_roberta_politeness
#SBATCH --output="/ukp-storage-1/nothvogel/gyafc-classifier/logs/log_politeness.txt"
#SBATCH --partition=cai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -d singleton
#SBATCH --ntasks=1
#SBATCH --mail-user=phil-n@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --account=cai-student

# Experiment Details :- RoBERTa model for politeness.
# Run Details :- none

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
conda activate trans
export WANDB_API_KEY=yourApiKey
export WANDB_CONFIG_DIR=/ukp-storage-1/nothvogel/.config/wandb

export TORCH_HOME=/ukp-storage-1/nothvogel/.cache/torch
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/ukp-storage-1/nothvogel/style-dataset-preprocessing/data/politeness"

# replace politeness with name of dataset!
# also replace labels and max epochs!

MAX_EPOCHS=3

SCRIPT_PATH="/ukp-storage-1/nothvogel/gyafc-classifier/train_classifier.py"

for lr in 1e-4 5e-5 1e-5; do
    for batchSize in 16 32; do
        python3 "$SCRIPT_PATH" \
            --org_data_dir "$DATA_DIR" \
            --data_set "politeness" \
            --label_0 "polite" \
            --label_1 "impolite" \
            --eval_batch_size 512 \
            --freeze_last_layers 0 \
            --train_batch_size $batchSize \
            --learning_rate $lr \
            --max_epochs $MAX_EPOCHS
    done
done
