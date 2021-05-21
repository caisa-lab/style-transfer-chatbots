import torch
import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import AdamW, Trainer, TrainingArguments
from scipy.special import softmax
from train_classifier import StyleClassificationDataset


def createReceptivenessDataset(df, tokenizer):
    # prepare dataset
    valTexts = df['text'].tolist()
    valLabels = [0 for _ in range(len(valTexts))]

    valEncodings = tokenizer(valTexts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True, max_length=256)
    valDataset = StyleClassificationDataset(valEncodings, valLabels)

    return valDataset

if __name__ == '__main__':
    # train model
    def predict():
        # important settings
        PRED_SOURCE_PATH = '/data/daten/datasets/receptiveness/flekSet.csv'
        modelName = 'roberta-base'
        modelDir = 'results/receptiveness/lr-0.0001_batch-16_layer-13'
        batchSize = 512
        allLabels = [
            'receptive',
            'not_receptive'
        ]

        outputDir = os.path.join(modelDir, 'pred')
        os.makedirs(outputDir, exist_ok=True)

        trainingArgs = TrainingArguments(
            output_dir='results/tmp_trainer',
            per_device_eval_batch_size=batchSize,   # batch size for evaluation
            dataloader_num_workers=4,
        )

        checkpointDir = os.path.join(modelDir, 'best')
        model = RobertaForSequenceClassification.from_pretrained(checkpointDir)
        tokenizer = RobertaTokenizer.from_pretrained(modelName)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.eval()
        trainer = Trainer(
            model=model,
            args=trainingArgs  
        )

        df = pd.read_csv(PRED_SOURCE_PATH)
        predDataset = createReceptivenessDataset(df, tokenizer)

        with torch.no_grad():
            preds = trainer.predict(predDataset)

        preds = preds.predictions
        # 0 = index of receptive label
        #labels = [allLabels[i] for i in preds.argmax(-1)]
        preds = softmax(preds, axis=1)
        receptiveAllPred = preds[:, 0]
        df = df.assign(receptiveAllPred=receptiveAllPred)
        df.to_csv(os.path.join(outputDir, 'result.csv'))
        
    predict()