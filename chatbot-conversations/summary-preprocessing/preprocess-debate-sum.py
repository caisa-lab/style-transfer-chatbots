# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import pandas as pd
import numpy as np
import os
import re

from IPython.display import display

SEED = 1337

# %%

DEBATE_SUM_PATH = '/data/daten/datasets/summary/debate sum/debateall.csv'
df = pd.read_csv(DEBATE_SUM_PATH)
# %%

dataSample = df.sample(3)[['Abstract', 'Extract', 'Full-Document']]
for _, sample in dataSample.iterrows():
    print('Abstract:')
    print(sample['Abstract'])
    print('Extract:')
    print(sample['Extract'][:200])
    print('Full Document:')
    print(sample['Full-Document'][:400])
    print('=======================')

longAbs = df.loc[df['#CharsAbstract'] > 800]
print('Number of samples with > 800 characters abs:', len(longAbs))
display(longAbs.sample(3))

for _, sample in longAbs.sample(3).iterrows():
    print('Abstract:')
    print(sample['Abstract'])
    print('Extract:')
    print(sample['Extract'][:200])
    print('Full Document:')
    print(sample['Full-Document'][:400])
    print('=======================')

# %%

numOfSamples = len(df)
df = df.loc[(df['#WordsAbstract'] <= 256) & (df['#WordsAbstract'] >= 3) & (df['#CharsAbstract'] <= 800)]
df = df.loc[(df['AbsCompressionRatio'] >= 0.11) & (df['AbsCompressionRatio'] <= 0.33)]
diffOfSamples = numOfSamples - len(df)
print('Discarded {} samples'.format(diffOfSamples))
display(df.describe())

def getBins(df, column, binSize):
    maxVal = df[column].max()
    return list(range(0, maxVal, binSize))

df.hist('#CharsAbstract', bins=getBins(df, '#CharsAbstract', 25))
df.hist('#WordsAbstract', bins=getBins(df, '#WordsAbstract', 5))
df.hist('AbsCompressionRatio')
# %%

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    validate = df.loc[perm[train_end:validate_end]]
    test = df.loc[perm[validate_end:]]
    return train, validate, test

def saveDf(df, splitName):
    outDir = 'data/debate-sum'
    os.makedirs(outDir, exist_ok=True)
    savePath = os.path.join(outDir, splitName + '.csv')
    df.to_csv(savePath)

df = df.assign(abstract=df['Abstract'], document=df['Full-Document'], absCompressionRatio=df['AbsCompressionRatio'])
df = df[['abstract', 'document', 'absCompressionRatio']]

WHITESPACE = re.compile(r'\s+')
def removeFormatting(text):
    text = text.replace('¶', '')
    text = text.replace('\n', ' <n> ').replace('\r', '')
    text = WHITESPACE.sub(' ', text)
    return text

df['document'] = df['document'].apply(removeFormatting)
df['abstract'] = df['abstract'].apply(removeFormatting)
trainDf, valDf, testDf = train_validate_test_split(df, train_percent=0.8, validate_percent=0.1, seed=SEED)

saveDf(trainDf, 'train')
saveDf(valDf, 'val')
saveDf(testDf, 'test')

# %%
