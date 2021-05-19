from collections import defaultdict
import os
import pandas as pd
import jsonpickle
from hit import Hit
from hit_list import orderHitsById

scenarioName = 'debate'
OUTPUT_DIR = os.path.join('plots', '{}-conversations'.format(scenarioName))

runName = '{}-conversations-1-live'.format(scenarioName)
df = pd.read_csv('hit-output/{}.assignments.csv'.format(runName), index_col=0)
df = df.loc[df['hitType'] == 'task']
df = df.loc[df['error'] != True]
df = df.loc[pd.isna(df['error'])]

with open('hit-data/{}.json'.format(runName), 'r') as f:
    hits = jsonpickle.decode(f.read(), keys=True, classes=Hit)
hitIndex = orderHitsById(hits)

# get flagged messages
redFlags = []
badHits = defaultdict(int)
for idx, row in df.iterrows():
    hit = hitIndex[row['hitId']]
    turnCount = len(hit.getBotTurns(isConst=False))
    for styleIndex in [1, 2]:
        styleName = row['style{}'.format(styleIndex)]
        for flag in ['Offensive[]', 'Incoherent[]']:
            flagColumn = 'bot{}{}'.format(styleIndex, flag)

            flagValue = row[flagColumn]
            if (pd.notna(flagValue) and len(flagValue) > 0):
                flags = flagValue.split('|')
                
                for turnId in flags:
                    redFlags.append({
                        'style': styleName,
                        'hitId': row['hitId'],
                        'turnId': turnId,
                        'flag': flag.replace('[]', '')
                    })
                    badHits[row['hitId']] += 1

redFlags = pd.DataFrame(redFlags)

badHitList = sorted(badHits.items(), key=lambda x: x[1], reverse=True)

print(badHitList[3])


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

currentTask = 'style'
with open(os.path.join(dname, 'templates', 'hit', currentTask, 'hit.html'), 'r') as f:
    fileTemplate = f.read()
with open(os.path.join(dname, 'templates', 'hit', currentTask, 'turn.html'), 'r') as f:
    turnTemplate = f.read()
with open(os.path.join(dname, 'templates', 'question.xml')) as f:
    questionTemplate = f.read()

for idx, (hitId, flagCount) in enumerate(badHitList[:10]):
    hit = hitIndex[hitId]
    html = hit.generateHtmlQuestion(fileTemplate, turnTemplate)
    for part in questionTemplate.split('%htmlContent%'):
        html = html.replace(part, '')
    df = redFlags
    style = df.loc[df['hitId'] == hitId].iloc[0]['style']
    targetPath = 'temp/bad-hits/hit-{}-{}-{}.html'.format(idx, style, flagCount)
    with open(targetPath, 'w') as f:
        f.write(html)