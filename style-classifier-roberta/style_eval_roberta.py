import torch
import os
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from train_classifier import StyleClassificationDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval_batch_size', type=int, default=128)
parser.add_argument('--input_txt', type=str, required=True)
parser.add_argument('--output_txt', type=str, required=True)
parser.add_argument('--checkpoint_dir', type=str, required=True)
parser.add_argument('--label_0', type=str, required=True)
parser.add_argument('--label_1', type=str, required=True)
parser.add_argument('--expected_label', type=str, required=True)
parser.add_argument('--model_name', type=str, default='roberta-base')

args = parser.parse_args()
print(args)

def computeMetrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def labelToNumber(currentLabel: str, allLabels: list):
    #label = np.zeros(len(allLabels))
    #label[allLabels.index(currentLabel)] = 1
    return allLabels.index(currentLabel)

def labelNumberToString(labelNumber, allLabels):
    #return allLabels[np.argmax(labelNumber)]
    return allLabels[labelNumber]

def readSplit(csvFilePath, allLabels: list):
    # my split is already preprocessed and does not need to be shuffled
    df = pd.read_csv(csvFilePath)

    texts = []
    labels = []
    for index, row in df.iterrows():
        texts.append(row['text'])
        labels.append(labelToNumber(row['label'], allLabels))
    
    return texts, labels


if __name__ == '__main__':
    # train model
    def predict():
        # important settings
        modelName = args.model_name
        checkpointDir = args.checkpoint_dir
        batchSize = args.eval_batch_size
        allLabels = [
            args.label_0,
            args.label_1
        ]

        def createDataset(txtPath, tokenizer, expectedLabel):
            # prepare dataset
            with open(txtPath, 'r') as f:
                testTexts = f.read().strip().splitlines()

            expectedIdx = labelToNumber(expectedLabel, allLabels)
            testLabels = [expectedIdx for _ in range(len(testTexts))]

            def tokenizeTexts(texts):
                return tokenizer(texts, return_tensors='pt', add_special_tokens=True, truncation=True, padding='max_length', max_length=256)
            
            testEncodings = tokenizeTexts(testTexts)

            testDataset = StyleClassificationDataset(testEncodings, testLabels)

            print('Test dataset examples:')
            print(testTexts[0] + ' == ' + labelNumberToString(testLabels[0], allLabels))
            print(testTexts[-1] + ' == ' + labelNumberToString(testLabels[-1], allLabels))

            return testDataset

        outputDir = os.path.join('results', 'tmp_trainer')
        os.makedirs(outputDir, exist_ok=True)

        trainingArgs = TrainingArguments(
            output_dir='results/tmp_trainer',
            per_device_eval_batch_size=batchSize,   # batch size for evaluation
            dataloader_num_workers=4,
        )

        model = RobertaForSequenceClassification.from_pretrained(checkpointDir)
        tokenizer = RobertaTokenizer.from_pretrained(modelName)
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.eval()
        trainer = Trainer(
            model=model,
            args=trainingArgs,
            compute_metrics=computeMetrics  
        )

        expectedLabel = args.expected_label
        predDataset = createDataset(args.input_txt, tokenizer, expectedLabel)

        with torch.no_grad():
            preds = trainer.predict(predDataset)

        print(preds.metrics)
        
        preds = preds.predictions
        # 0 = index of receptive label
        #labels = [allLabels[i] for i in preds.argmax(-1)]
        preds = softmax(preds, axis=1)

        expectedIdx = labelToNumber(expectedLabel, allLabels)
        print(preds[:3])
        predIndices = preds.argmax(-1)
        assert(len(preds) == len(predIndices))
        print(predIndices[:3])

        # expected output format: correct/incorrect, expectedLabel, predictedLabel
        outputs = []
        for predIdx in predIndices:
            correctLabel = 'correct' if (predIdx == expectedIdx) else 'incorrect'
            predictedLabel = labelNumberToString(predIdx, allLabels)
            output = '{},{},{}'.format(correctLabel, expectedLabel, predictedLabel)
            outputs.append(output)

        with open(args.output_txt, 'w') as f:
            f.write('\n'.join(outputs))
        
        print('Wrote output to:', args.output_txt)
        
    predict()