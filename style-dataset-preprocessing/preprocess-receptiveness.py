from utils import setSeed, printSplitStats, balanceDataset
import os
import pandas as pd
import numpy as np
from wordsegment import load, segment
from nltk.tokenize import sent_tokenize
from collections import defaultdict

setSeed()

def preprocessReceptivenessV3():
    RECEPTIVENESS_PATH = 'data/receptiveness'

    df = pd.read_csv(os.path.join(RECEPTIVENESS_PATH, 'source.csv'), index_col=0)

    # split up paragraphs to sentences
    sentenceDict = defaultdict(dict)
    newDf = []
    for index, row in df.iterrows():
        label = row['label']
        for sentence in sent_tokenize(row['text']):
            if ((len(sentence) <= 5) or sentence in sentenceDict[label]):
                continue
            newRow = row.to_dict()
            newRow['text'] = sentence
            sentenceDict[label][sentence] = True
            newDf.append(newRow)

    df = pd.DataFrame(newDf)
    df = df.assign(id=list(range(len(df))))
    print(df['label'].value_counts())
    print('total samples:', len(df))
    
    df = df.sample(frac=1, replace=False)
    dfReceptive = df.loc[df['label'] == 'receptive']
    dfNotReceptive = df.loc[df['label'] != 'receptive']
    numOfSamples = np.array((len(dfReceptive), len(dfNotReceptive)))

    splits = {
        'train': 0.8 * numOfSamples,
        'val': 0.1 * numOfSamples,
        'test': 0.1 * numOfSamples
    }
    targetDir = os.path.join('data', 'receptiveness')
    os.makedirs(targetDir, exist_ok=True)
    iterationCtr = 1
    dfList = []
    for split, (numReceptive, numNotReceptive) in splits.items():
        if (iterationCtr == len(splits.keys())):
            # use remaining samples
            currentReceptive = dfReceptive
            currentNotReceptive = dfNotReceptive
        else:
            currentReceptive = dfReceptive.sample(int(numReceptive), replace=False)
            dfReceptive = dfReceptive.append(currentReceptive).drop_duplicates(keep=False)
            
            currentNotReceptive = dfNotReceptive.sample(int(numNotReceptive), replace=False)
            dfNotReceptive = dfNotReceptive.append(currentNotReceptive).drop_duplicates(keep=False)

        # combine dfs and shuffle
        currentDf = currentReceptive.append(currentNotReceptive).sample(frac=1, replace=False)
        currentDf.to_csv(os.path.join(targetDir, split + '.csv'))
        printSplitStats(currentDf, 'style ' + split)

        currentDf = currentDf.loc[pd.isna(currentDf['receptiveAllPred'])]
        dfList.append((split, currentDf))
        iterationCtr += 1

    for split, currentDf in dfList:
        currentDf = balanceDataset(currentDf)
        currentDf.to_csv(os.path.join(targetDir, 'classification', split + '.csv'))
        printSplitStats(currentDf, 'classification ' + split)

preprocessReceptivenessV3()