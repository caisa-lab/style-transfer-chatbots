#!/bin/sh
#SBATCH --job-name=finetune_gpt2_formality_1
#SBATCH --output="/ukp-storage-1/nothvogel/style-transfer-paraphrase/style_paraphrase/logs/log_formality_1_ckpts.txt"
#SBATCH --partition=cai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40GB
#SBATCH -d singleton
#SBATCH --ntasks=1
#SBATCH --mail-user=p.nothvogel@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=cai-student

# Experiment Details :- GPT2 model for formality.
# Run Details :- accumulation = 2, batch_size = 5, beam_size = 1, cpus = 3, dataset = datasets/formality, eval_batch_size = 1, global_dense_feature_list = none, gpu = m40, learning_rate = 5e-5, memory = 50, model_name = gpt2, ngpus = 1, num_epochs = 3, optimizer = adam, prefix_input_type = paraphrase_250, save_steps = 500, save_total_limit = -1, specific_style_train = 0, stop_token = eos

export DATA_DIR=datasets/formality

BASE_DIR=style_paraphrase

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


python3 $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/model_formality_1 \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_eval \
    --do_delete_old \
    --save_total_limit 4 \
    --data_dir=$DATA_DIR \
    --save_steps 500 \
    --logging_steps 20 \
    --evaluate_during_training \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --per_gpu_train_batch_size 32 \
    --job_id formality_1 \
    --learning_rate 5e-5 \
    --prefix_input_type paraphrase_250 \
    --global_dense_feature_list none \
    --specific_style_train 1 \
    --optimizer adam

