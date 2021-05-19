import argparse
import wandb
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

parser = argparse.ArgumentParser()
parser.add_argument('--org_data_dir', type=str, required=True)
parser.add_argument('--data_set', type=str, required=True)
parser.add_argument('--label_0', type=str, required=True)
parser.add_argument('--label_1', type=str, required=True)

args = parser.parse_args()

runName = 'majority'
projectName = 'final-{}-regression'.format(args.data_set)

wandb.init(project=projectName, name=runName, entity='philno')
wandb.config.update(args)

def labelToNumber(currentLabel: str, allLabels: list):
    #label = np.zeros(len(allLabels))
    #label[allLabels.index(currentLabel)] = 1
    return allLabels.index(currentLabel)

def computeMetrics(trueLabels, preds, prefix=''):
    precision, recall, f1, _ = precision_recall_fscore_support(trueLabels, preds, average='weighted')
    acc = accuracy_score(trueLabels, preds)
    return {
        prefix + 'accuracy': acc,
        prefix + 'f1': f1,
        prefix + 'precision': precision,
        prefix + 'recall': recall
    }

def readSplit(csvFilePath, allLabels: list):
    # my split is already preprocessed and does not need to be shuffled
    df = pd.read_csv(csvFilePath)

    texts = []
    labels = []
    for index, row in df.iterrows():
        texts.append(row['text'])
        labels.append(labelToNumber(row['label'], allLabels))
    
    return texts, labels

ORG_DATA_DIR = args.org_data_dir
allLabels = [
    args.label_0,
    args.label_1
]

df = pd.read_csv(os.path.join(ORG_DATA_DIR, 'train.csv'), index_col=0)
majorityLabel = str(df['label'].value_counts(sort=True).index[0])
majorityLabel = labelToNumber(majorityLabel, allLabels)

trainTexts, trainLabels = readSplit(os.path.join(ORG_DATA_DIR, 'train.csv'), allLabels)
valTexts, valLabels = readSplit(os.path.join(ORG_DATA_DIR, 'val.csv'), allLabels)
testTexts, testLabels = readSplit(os.path.join(ORG_DATA_DIR, 'test.csv'), allLabels)

trainLabels = np.array(trainLabels, dtype=np.int)
trainPreds = np.array([majorityLabel for i in range(len(trainLabels))], dtype=np.int)
trainMetrics = computeMetrics(trainLabels, trainPreds, prefix='train/')
print('train metrics:')
print(trainMetrics)
print('======================')

valLabels = np.array(valLabels, dtype=np.int)
valPreds = np.array([majorityLabel for i in range(len(valLabels))], dtype=np.int)
valMetrics = computeMetrics(valLabels, valPreds, prefix='val/')
print('val metrics:')
print(valMetrics)

testLabels = np.array(testLabels, dtype=np.int)
testPreds = np.array([majorityLabel for i in range(len(testLabels))], dtype=np.int)
testMetrics = computeMetrics(testLabels, testPreds, prefix='test/')
print('test metrics:')
print(testMetrics)

wandb.log(trainMetrics)
wandb.log(valMetrics)
wandb.log(testMetrics)
wandb.run.summary.update(valMetrics)
wandb.run.summary.update(testMetrics)