from utils import setSeed, printSplitStats
import numpy as np
import pandas as pd
import os
import random

setSeed()

INPUT_PATH = 'data/politeness/raw'
OUTPUT_PATH = 'data/politeness'
os.makedirs(os.path.join(OUTPUT_PATH), exist_ok=True)

def renameTextCol(tempDf):
    tempDf = tempDf.rename(columns={ 'txt': 'text' })
    return tempDf

df = pd.read_csv(os.path.join(INPUT_PATH, 'politeness.csv'))
df = df.loc[df['is_useful'] == 1]
df = df.loc[(df['p_tag'] == 'P_9') | (df['p_tag'] == 'P_0') | (df['p_tag'] == 'P_1')]

labels = pd.Series(index=df.index)
labels[df['p_tag'] == 'P_9'] = 'polite'
labels[df['p_tag'] == 'P_0'] = 'impolite'
df = df.assign(label=labels)
df = renameTextCol(df)

trainDf = df.loc[df['split'] == 'train']
valDf = df.loc[df['split'] == 'val']
testDf = df.loc[df['split'] == 'test']

trainSize = 50000
valSize = int(0.1 * trainSize)
testSize = valSize

trainPolite = trainDf.loc[trainDf['label'] == 'polite'].sample(trainSize, replace=False)
trainImpolite = trainDf.loc[trainDf['label'] == 'impolite'].sample(trainSize, replace=False)
# append dfs and shuffle
trainDf = trainPolite.append(trainImpolite).sample(frac=1)

valPolite = valDf.loc[valDf['label'] == 'polite'].sample(valSize, replace=False)
valImpolite = valDf.loc[valDf['label'] == 'impolite'].sample(valSize, replace=False)
valDf = valPolite.append(valImpolite).sample(frac=1)

# not sure what test df should be.
testPolite = testDf.loc[testDf['label'] == 'polite'].sample(testSize, replace=False)
testImpolite = testDf.loc[testDf['label'] == 'impolite'].sample(testSize, replace=False)
testDf = testPolite.append(testImpolite).sample(frac=1)

trainDf.to_csv(os.path.join(OUTPUT_PATH, 'train.csv'))
valDf.to_csv(os.path.join(OUTPUT_PATH, 'val.csv'))
testDf.to_csv(os.path.join(OUTPUT_PATH, 'test.csv'))
printSplitStats(trainDf, 'train')
printSplitStats(valDf, 'val')
printSplitStats(testDf, 'test')

def saveSplit(df, filePath):
    texts = df['txt']
    labels = df['label']

    with open(filePath + '.txt', 'w') as f:
        f.writelines(t + '\n' for t in texts)
    with open(filePath + '.label', 'w') as f:
        f.writelines(l + '\n' for l in labels)

# format for paraphrasing
#saveSplit(trainDf, os.path.join(DATA_PATH, 'splits', 'train'))
#saveSplit(valDf, os.path.join(DATA_PATH, 'splits', 'dev'))
#saveSplit(testDf, os.path.join(DATA_PATH, 'splits', 'test'))