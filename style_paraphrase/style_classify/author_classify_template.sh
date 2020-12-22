#!/bin/sh
#SBATCH --job-name=style_classify_{job_id}
#SBATCH --output="/ukp-storage-1/nothvogel/style-transfer-paraphrase/style_paraphrase/style_classify/logs/log_{job_id}.txt"
#SBATCH --partition=aiphes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40GB
#SBATCH -d singleton
#SBATCH --ntasks=1
#SBATCH --mail-user=p.nothvogel@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=aiphes-student

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

ROBERTA_LARGE="/ukp-storage-1/nothvogel/.cache/torch/roberta.large"

TOTAL_NUM_UPDATES={total_updates}
WARMUP_UPDATES={warmup}
LR={learning_rate}
NUM_CLASSES={num_classes}
MAX_SENTENCES={max_sentences}
ROBERTA_MODEL={roberta_model}
MAX_POSITIONS={max_positions}
UPDATE_FREQ={update_freq}


if [ "$ROBERTA_MODEL" = "LARGE" ]; then
   ROBERTA_PATH=$ROBERTA_LARGE/model.pt
   ROBERTA_ARCH=roberta_large
else
   ROBERTA_PATH=$ROBERTA_BASE/model.pt
   ROBERTA_ARCH=roberta_base
fi


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
conda activate style
export WANDB_API_KEY=6de83da6c6fa47080f927222261e75c1d7c8bf01
export WANDB_CONFIG_DIR=/ukp-storage-1/nothvogel/.config/wandb

python3 fairseq/train.py {base_dataset}/{dataset}-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions $MAX_POSITIONS \
    --no-epoch-checkpoints \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch $ROBERTA_ARCH \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch {num_epochs} \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --find-unused-parameters \
    --update-freq $UPDATE_FREQ \
    --save-dir style_paraphrase/style_classify/saved_models/save_{job_id} \
    --log-interval 100 \
    --wandb-project fairseq_cds