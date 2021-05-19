import os
import re
import pandas as pd

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
dataDir = 'debate-conversations-1'
inputDf = pd.read_csv(os.path.join(dname, dataDir, 'input.csv'))
paraDf = pd.read_csv(os.path.join(dname, dataDir, 'paraphrases.csv'), index_col=0)
output = []

def getLengthRatio(row):
    orgLength = float(len(row['original_sentence']))
    genLength = float(len(row['style_transfer']))
    return genLength / orgLength

lengthRatios = []
for idx, row in paraDf.iterrows():
    lengthRatios.append(getLengthRatio(row))
paraDf = paraDf.assign(length_ratio=lengthRatios)

punctRegex = re.compile('[?.!:]$')
# select candidates based on perplexity
for style in paraDf['target_style'].unique():
    tempDf = []
    df = paraDf
    df = df.loc[df['target_style'] == style]
    df = df.loc[df['length_ratio'] <= 2.5]
    for sentenceIdx in df['sentence_idx'].unique():
        candidates = df.loc[df['sentence_idx'] == sentenceIdx]
        candidates = candidates.sort_values(by='style_perplexity')
        bestCandidate = candidates.iloc[0].to_dict()
        tempDf.append(bestCandidate)

    tempDf = pd.DataFrame(tempDf)
    # match up input index with generated sentences
    for sourceIdx, row in inputDf.iterrows():
        row = row.to_dict()
        # only use paraphrase of rows that are not const
        if (not row['isConst']):
            df = tempDf
            df = df.loc[df['source_idx'] == sourceIdx].sort_values(by='sentence_idx')
            sentences = []
            for sentence in df['style_transfer']:
                sentence = sentence.strip()
                if (not punctRegex.search(sentence)):
                    sentence += '.'
                sentences.append(sentence)

            row['text'] = ' '.join(sentences).strip().replace('@-@', '-')

        row['targetStyle'] = style
        output.append(row)

output = pd.concat([inputDf, pd.DataFrame(output)], ignore_index=True)
output = output.loc[:, ~output.columns.str.contains('^Unnamed')]
output.to_csv(os.path.join(dname, dataDir, 'output.csv'))
