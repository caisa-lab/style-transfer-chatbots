from utils import setSeed, printSplitStats
import os
import pandas as pd

setSeed()

# convert the original split of krishna et al

def readGyafcSplit(splitName):
    GYAFC_PATH = 'data/gyafc/raw/'

    def readLines(filePath):
        with open(filePath, 'r') as f:
            texts = f.read().splitlines()

        return [l for l in texts if len(l) > 0]

    text0 = readLines(os.path.join(GYAFC_PATH, splitName + '_0.txt'))
    attr0 = readLines(os.path.join(GYAFC_PATH, splitName + '_0.attr'))
    attr0 = attr0[:len(text0)]

    text1 = readLines(os.path.join(GYAFC_PATH, splitName + '_1.txt'))
    attr1 = readLines(os.path.join(GYAFC_PATH, splitName + '_1.attr'))
    attr1 = attr1[:len(text1)]

    texts = text0 + text1
    labels = attr0 + attr1

    newDf = {
        'id': list(range(len(texts))),
        'text': texts,
        'label': labels
    }
    df = pd.DataFrame(newDf)
    return df

def preprocessGyafc():
    trainDf = readGyafcSplit('train')
    # shuffle training data
    trainDf = trainDf.sample(frac=1, replace=False)
    valDf = readGyafcSplit('dev')
    testDf = readGyafcSplit('test')

    OUT_PATH = 'data/gyafc'
    trainDf.to_csv(os.path.join(OUT_PATH, 'train.csv'))
    valDf.to_csv(os.path.join(OUT_PATH, 'val.csv'))
    testDf.to_csv(os.path.join(OUT_PATH, 'test.csv'))
    printSplitStats(trainDf, 'train')
    printSplitStats(valDf, 'val')
    printSplitStats(testDf, 'test')

preprocessGyafc()
