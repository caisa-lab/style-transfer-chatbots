import pandas as pd
import numpy as np
import os
from nltk.tokenize import sent_tokenize

def getSummaries():
    PROD_PATH = '/data/daten/datasets/summary/products business/Copycat-abstractive-opinion-summarizer/gold_summs/all.csv'
    df = pd.read_csv(PROD_PATH, sep='\t')

    summ = df[['summ1', 'summ2', 'summ3']]
    summ = summ.values.reshape(-1)
    return summ

def getReceptiveness():
    RECEP_PATH = '/data/daten/datasets/receptiveness/flekSet.csv'
    df = pd.read_csv(RECEP_PATH)
    df = df.sample(100, replace=False)

    return df['text']

def getOffReviews():
    OFF_PATH = '/data/daten/python/master/gyafc-classifier/offensive-reviews-roberta.csv'
    df = pd.read_csv(OFF_PATH)
    df = df.sample(300, replace=False)

    return df['summary']

SEED = 1337
np.random.seed(SEED)
#outFileName = 'prod-summaries.txt'
#paragraphs = getSummaries()
#outFileName = 'receptiveness.txt'
#paragraphs = getReceptiveness()
outFileName = 'off-reviews.txt'
paragraphs = getOffReviews()
sentences = []
for paragraph in paragraphs:
    sentences += sent_tokenize(paragraph)

sentences = [s.replace('\n', ' ').replace('\r', '') for s in sentences[:] if (len(s) > 6)]
with open(os.path.join('samples', 'data_samples', outFileName), 'w') as f:
    f.write('\n'.join(sentences))