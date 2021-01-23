#!/bin/sh
#SBATCH --job-name=convert_dataset
#SBATCH --output="/ukp-storage-1/nothvogel/style-transfer-paraphrase/convert_dataset_log.txt"
#SBATCH --partition=cai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH -d singleton
#SBATCH --ntasks=1
#SBATCH --mail-user=p.nothvogel@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=cai-student

# Experiment Details :- GPT2 model for style transfer.
# Run Details :- Converting dataset to bpe.



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
export ROBERTA_LARGE="/ukp-storage-1/nothvogel/.cache/huggingface/transformers/roberta.large"


DATASET=datasets/receptiveness
python3 datasets/dataset2bpe.py --dataset $DATASET

sh datasets/bpe2binary.sh $DATASET

python3 datasets/paraphrase_splits.py --dataset $DATASET --model_dir man-generation/models/paraphraser_gpt2_large