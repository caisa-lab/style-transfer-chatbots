#!/bin/sh
#SBATCH --job-name=finetune_gpt2_{job_id}
#SBATCH --output="/ukp-storage-1/nothvogel/style-transfer-paraphrase/style_paraphrase/logs/log_{job_id}.txt"
#SBATCH --partition=aiphes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem=40GB
#SBATCH -d singleton
#SBATCH --ntasks=1
#SBATCH --mail-user=p.nothvogel@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=aiphes-student

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

export DATA_DIR={dataset}

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
    --output_dir=$BASE_DIR/saved_models/model_{job_id} \
    --model_type=gpt2 \
    --model_name_or_path={model_name} \
    --do_train \
    --data_dir=$DATA_DIR \
    --save_steps {save_steps} \
    --logging_steps 20 \
    --do_delete_old \
    --save_total_limit 3 \
    --evaluate_during_training \
    --num_train_epochs {num_epochs} \
    --gradient_accumulation_steps {accumulation} \
    --per_gpu_train_batch_size {batch_size} \
    --job_id {job_id} \
    --learning_rate {learning_rate} \
    --prefix_input_type {prefix_input_type} \
    --global_dense_feature_list {global_dense_feature_list} \
    --specific_style_train {specific_style_train} \
    --optimizer {optimizer}
