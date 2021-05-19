# Understanding appropriate chatbot style: 
## Summarization and stylistic paraphrasing of opinionated text to generate chatbot answers

This repository contains all the code related to the Master thesis of Philipp Nothvogel: 
`Understanding appropriate chatbot style: Summarization and stylistic paraphrasing of opinionated text to generate chatbot answers`. It was completed in May 2021.

## Repository structure

The code is structured as follows:

* `style-dataset-preprocessing` contains all preprocessing steps to form our style corpora.
* `style-classifier-baselines` contains the style classifier regression and majority baselines.
* `style-classifier-roberta` contains the code for training the RoBERTa style classifiers which we use to evaluate the stylistic paraphrasers.
* `style-transfer-paraphrase` contains the stylistic paraphrasing code which we adapted from [STRAP](https://github.com/martiansideofthemoon/style-transfer-paraphrase).
* `chatbot-conversations` contains all code related to generating chatbot answers and conversations from opinionated text.
* `human-eval` contains code for the crowdsourcing study.

## General hints

We provide the exact versions of python packages that we used for the experiments in the `env.yaml` files. 
You can use the `conda env create -f env.yaml` command to create the same environment on your machine. 

The code should mostly work out of the box. 
You will need to adjust the paths to the datasets in python or shell scripts.
Note that we cannot upload most of the datasets to git due to licensing issues.
However, you can obtain the original datasets yourself and use the code provided in `style-dataset-preprocessing` to reproduce our datasets and splits.
