from nltk.tokenize import WhitespaceTokenizer
from nltk.probability import FreqDist
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import os
import itertools
from nltk import ngrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack, csr_matrix
import wandb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=str, required=True)
parser.add_argument('--org_data_dir', type=str, required=True)
parser.add_argument('--label_0', type=str, required=True)
parser.add_argument('--label_1', type=str, required=True)
parser.add_argument('--n_gram', type=int, required=True)
parser.add_argument('--token_level', type=str, required=True)

args = parser.parse_args()

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
nGram = args.n_gram
tokenLevel = args.token_level
runName = '{}_{}-gram'.format(tokenLevel, nGram)
projectName = 'final-{}-regression'.format(args.data_set)

wandb.init(project=projectName, name=runName, entity='philno')
wandb.config.update(args)

trainTexts, trainLabels = readSplit(os.path.join(ORG_DATA_DIR, 'train.csv'), allLabels)
valTexts, valLabels = readSplit(os.path.join(ORG_DATA_DIR, 'val.csv'), allLabels)
testTexts, testLabels = readSplit(os.path.join(ORG_DATA_DIR, 'test.csv'), allLabels)

tokenizer = WhitespaceTokenizer()

def tokenize(texts):
    tokens = []
    if (tokenLevel == 'words'):
        texts = tokenizer.tokenize_sents(texts)

    for sentence in texts:
        if (nGram > 1): 
            tokens.append(list(ngrams(sentence, nGram)))
        else:
            tokens.append(sentence)
    return tokens

tokens = itertools.chain(*tokenize(trainTexts))
fDist = FreqDist(tokens)
vocabSize = len(fDist)
print('Vocabulary size:', vocabSize)
vocabSize = min(vocabSize, 25000)
print('Vocabulary size:', vocabSize)


tokenToIdx = {}
for idx, (token, count) in enumerate(fDist.most_common(vocabSize)):
    tokenToIdx[token] = idx

def encodeTexts(texts):
    encodings = []
    for tokens in tokenize(texts):
        encoding = np.zeros((vocabSize,), dtype=np.int)
        for t in tokens:
            if (t in tokenToIdx):
                encoding[tokenToIdx[t]] = 1
        encodings.append(csr_matrix(encoding))
    
    encodings = vstack(encodings)
    return encodings

trainEncodings = encodeTexts(trainTexts)
trainLabels = np.array(trainLabels, dtype=np.int)

model = LogisticRegression(verbose=0)
model.fit(trainEncodings, trainLabels)
trainPreds = model.predict(trainEncodings)
trainMetrics = computeMetrics(trainLabels, trainPreds, prefix='train/')
print('train metrics:')
print(trainMetrics)
print('======================')

valEncodings = encodeTexts(valTexts)
valLabels = np.array(valLabels, dtype=np.int)
valPreds = model.predict(valEncodings)
valMetrics = computeMetrics(valLabels, valPreds, prefix='val/')
print('val metrics:')
print(valMetrics)

testEncodings = encodeTexts(testTexts)
testLabels = np.array(testLabels, dtype=np.int)
testPreds = model.predict(testEncodings)
testMetrics = computeMetrics(testLabels, testPreds, prefix='test/')
print('test metrics:')
print(testMetrics)

wandb.log(trainMetrics)
wandb.log(valMetrics)
wandb.log(testMetrics)
wandb.run.summary.update(valMetrics)
wandb.run.summary.update(testMetrics)

#wandb.sklearn.plot_regressor(model, trainEncodings, valEncodings, trainLabels, valLabels,  model_name=runName)
featureNames = list(tokenToIdx.items())
featureNames.sort(key=(lambda f: f[1]))
featureNames = [f[0] for f in featureNames[:]]

#wandb.sklearn.plot_classifier(model, trainEncodings, valEncodings, trainLabels, valLabels, valPreds,
#    model.predict_proba(valEncodings), allLabels, model_name=runName, feature_names=featureNames, is_binary=True)
wandb.sklearn.plot_confusion_matrix(testLabels, testPreds, allLabels)