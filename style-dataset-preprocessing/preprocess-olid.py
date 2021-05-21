from utils import balanceDataset, setSeed, printSplitStats
import os
import pandas as pd
import random
import pathlib
import re
import emoji
from wordsegment import load, segment
from nltk.tokenize import sent_tokenize

load()
setSeed()

hashtagRegex = re.compile(r'(#.+?)( |$)')
atUserRegex = re.compile(r'(@USER\s*){3,}')
whitespaceRegex = re.compile(r'\s+')
repeatedCharsRegex = re.compile(r'((.)\2{3,})')
punctuationRegex = re.compile(r'(([.,!?;])(?!\2| )(.))')
emailRegex = re.compile(r'[^@\s]+@[^@\s]+\.[^@\s]+')
urlRegex = re.compile(r'(https?:\/\/)?([\w\-])+\.{1}([a-zA-Z]{2,63})([\/\w-]*)*\/?\??([^#\n\r]*)?#?([^\n\r]*)')

def replaceEmojis(text):
    return emoji.demojize(text, delimiters=[' ', ' ']).replace('_', ' ')

def tokenizeHashtags(text):
    matches = hashtagRegex.findall(text)
    for match in matches:
        hashtag = match[0]
        if (len(hashtag) > 1):
            text = text.replace(hashtag, ' '.join(segment(hashtag)))
    text = text.replace('URL', 'http')
    text = atUserRegex.sub(' @USER @USER @USER ', text)
    text = whitespaceRegex.sub(' ', text)
    return text

def replaceRepeatedChars(text):
    matches = repeatedCharsRegex.findall(text)
    for match in matches:
        repeated = match[0]
        text = text.replace(repeated, 3 * match[1])
    
    return text

def fixPunctuationSpacing(text):
    matches = punctuationRegex.findall(text)
    for match in matches:
        fullMatch = match[0]
        replacement = '{} {}'.format(match[1], match[2])
        text = text.replace(fullMatch, replacement)
    
    return text

def replaceUrlsAndEmails(text):
    text = emailRegex.sub('EMAIL', text)
    text = urlRegex.sub('URL', text)
    return text

def processOlidTrainingData(forParaphrasing=False):
    OLID_PATH = 'data/olid/raw'

    newDf = []
    df = pd.read_csv(os.path.join(OLID_PATH, 'olid-training-v1.0.tsv'), sep='\t')
    #print(df)

    test = u'@USER @USER   @USER @USER @USER Obama   wanted liberals   &amp; illegal URL'
    test = replaceEmojis(test)
    test = tokenizeHashtags(test)

    # user mentions at the start of the text
    userMentions = re.compile(r'^(\s*@USER\s?)+')
    knownSentences = dict(OFF={}, NOT={})
    for index, row in df.iterrows():
        text = replaceEmojis(row['tweet'])
        text = tokenizeHashtags(text)
        label = row['subtask_a']

        if (forParaphrasing):
            # has to be one sentence per row/line
            for sentence in sent_tokenize(text):
                sentence = userMentions.sub('', sentence).strip()
                if (len(sentence) > 5 and sentence not in knownSentences[label]):
                    newDf.append({
                        'id': row['id'],
                        'text': sentence,
                        'label': label
                    })
                    knownSentences[label][sentence] = True
        else:
            newDf.append({
                'id': row['id'],
                'text': text,
                'label': label
            })
    
    random.shuffle(newDf)

    # generate splits
    newDf = pd.DataFrame(newDf)
    offensive = newDf.loc[newDf['label'] == 'OFF']
    notOffensive = newDf.loc[newDf['label'] == 'NOT']

    valSplit = 0.1
    numOfOffSamples = int(valSplit * float(len(offensive)))
    numOfNotSamples = int(valSplit * float(len(notOffensive)))

    valDf = pd.concat([offensive[:numOfOffSamples], notOffensive[:numOfNotSamples]])
    trainDf = pd.concat([offensive[numOfOffSamples:], notOffensive[numOfNotSamples:]])

    # shuffle train df
    trainDf = trainDf.sample(frac=1, replace=False)

    basePath = 'data/olid/'
    pathlib.Path(basePath).mkdir(parents=True, exist_ok=True)
    valDf.to_csv(os.path.join(basePath, 'val.csv'))
    trainDf.to_csv(os.path.join(basePath, 'train.csv'))

    printSplitStats(trainDf, 'style train:')
    printSplitStats(valDf, 'style val:')

    # downsample for classification
    basePath = 'data/olid/classification'
    pathlib.Path(basePath).mkdir(parents=True, exist_ok=True)
    valDf = balanceDataset(valDf)
    trainDf = balanceDataset(trainDf)
    valDf.to_csv(os.path.join(basePath, 'val.csv'))
    trainDf.to_csv(os.path.join(basePath, 'train.csv'))

    printSplitStats(trainDf, 'class train:')
    printSplitStats(valDf, 'class val:')



def processOlidTestData():
    OLID_PATH = 'data/olid/raw'

    textDf = pd.read_csv(os.path.join(OLID_PATH, 'testset-levela.tsv'), sep='\t')
    labelDf = pd.read_csv(os.path.join(OLID_PATH, 'labels-levela.csv'), names=['id', 'label'])
    df = textDf.assign(label=labelDf['label'])
    #print(df)
    newDf = []

    # user mentions at the start of the text
    userMentions = re.compile(r'^(\s*@USER\s?)+')
    knownSentences = dict(OFF={}, NOT={})
    for index, row in df.iterrows():
        text = replaceEmojis(row['tweet'])
        text = tokenizeHashtags(text)
        label = row['label']

        # has to be one sentence per row/line
        for sentence in sent_tokenize(text):
            sentence = userMentions.sub('', sentence).strip()
            if (len(sentence) > 5 and sentence not in knownSentences[label]):
                newDf.append({
                    'id': row['id'],
                    'text': sentence,
                    'label': label
                })
                knownSentences[label][sentence] = True

    testDf = pd.DataFrame(newDf)
    basePath = 'data/olid/'
    pathlib.Path(basePath).mkdir(parents=True, exist_ok=True)
    testDf.to_csv(os.path.join(basePath, 'test.csv'))
    printSplitStats(testDf, 'style test:')

    basePath = 'data/olid/classification'
    pathlib.Path(basePath).mkdir(parents=True, exist_ok=True)
    testDf = balanceDataset(testDf)
    testDf.to_csv(os.path.join(basePath, 'test.csv'))
    printSplitStats(testDf, 'class test:')

processOlidTrainingData(forParaphrasing=True)
processOlidTestData()