import os
import pandas as pd

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
inputDf = pd.read_csv(os.path.join(dname, 'input.csv'))
paraDf = pd.read_csv(os.path.join(dname, 'paraphrases.csv'), index_col=0)
output = []
# select candidates based on perplexity
for style in paraDf['target_style'].unique():
    tempDf = []
    df = paraDf
    df = df.loc[df['target_style'] == style]
    for sentenceIdx in df['sentence_idx'].unique():
        candidates = df.loc[df['sentence_idx'] == sentenceIdx]
        candidates = candidates.sort_values(by='style_perplexity')
        bestCandidate = candidates.iloc[0].to_dict()
        tempDf.append(bestCandidate)

    tempDf = pd.DataFrame(tempDf)
    # match up input index with generated sentences
    for sourceIdx in tempDf['source_idx'].unique():
        df = tempDf
        df = df.loc[df['source_idx'] == sourceIdx]
        row = inputDf.loc[sourceIdx].to_dict()
        row['text'] = ' '.join(df['style_transfer']).strip()
        row['targetStyle'] = style
        output.append(row)

output = inputDf.append(pd.DataFrame(output))
output.to_csv(os.path.join(dname, 'output.csv'))