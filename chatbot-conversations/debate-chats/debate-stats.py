from os import truncate
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/data/daten/datasets/receptiveness/s2convos.csv')

totalLens = []
isIncompletes = []
for idx, row in df.iterrows():
    totalLen = 0
    isIncomplete = False
    for column in '"round_1","round_2_part","round_3","round_4_part","round_5"'.split(','):
        column = column.replace('"', '')
        text = row[column]
        if (pd.notna(text) and len(text) >= 5):
            totalLen += len(text)
        else:
            isIncomplete = True
      
    totalLens.append(totalLen)
    isIncompletes.append(isIncomplete)

df = df.assign(totalLen=totalLens, isIncomplete=isIncompletes)
df = df.loc[~ df['isIncomplete']]
issues = df['issue'].unique()
pd.DataFrame({'issue': issues}).to_csv('issues.csv')
df = df.loc[df['totalLen'] <= 1600]
df = df.loc[df['totalLen'] >= 1000]

df['totalLen'].hist()
plt.show()
df.to_csv('short_candidates.csv')