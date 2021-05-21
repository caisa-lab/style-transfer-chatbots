from os import replace
import random
import numpy as np
import pandas as pd

def setSeed(SEED: int = 1337):
    random.seed(SEED)
    np.random.seed(SEED)

def printSplitStats(df, message, column='label'):
    print('============================')
    print(message)
    print('Num of samples:', len(df))
    print(df[column].value_counts())
    print(df[column].value_counts(normalize=True))

def balanceDataset(df):
    labelCount = df['label'].value_counts(sort=True)

    minLabel = str(labelCount.index[1])
    minCount = labelCount[1]
    minority = df.loc[df['label'] == minLabel]
    downSample = df.loc[df['label'] != minLabel].sample(minCount, replace=False)

    # concat and shuffle new df
    newDf = pd.concat([minority, downSample]).sample(frac=1, replace=False)
    return newDf