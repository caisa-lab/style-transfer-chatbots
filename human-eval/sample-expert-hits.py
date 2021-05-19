import pandas as pd
import jsonpickle
import os
from collections import defaultdict
from hit import Hit
from hit_list import orderHitsById

df = pd.read_csv('hit-output/debate-conversations-1-live.results.csv', index_col=0)
df = df.loc[df['hitType'] == 'task']
# shuffle df
df = df.sample(frac=1)

knownMatchups = defaultdict(int)

numOfSamples = 2
selectedHitIds = []
conversationIds = set()
for idx, row in df.iterrows():
    matchupList = sorted([row['style1'], row['style2']])
    matchup = str(matchupList)
    conversationId = row['conversationId']
    if (knownMatchups[matchup] >= numOfSamples or conversationId in conversationIds):
        continue
    knownMatchups[matchup] += 1
    selectedHitIds.append(row['hitId'])
    conversationIds.add(conversationId)

def createDirs(filePath: str):
    """Creates all missing directories for the given file path."""
    dirName = os.path.dirname(filePath)
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    return dirName

def saveHits(hits, runName):
    path = 'hit-data/{}.json'.format(runName)
    createDirs(path)
    with open(path, 'w') as f:
        jsonString = jsonpickle.encode(hits)
        f.write(jsonString)      
    print('Saved HIT data in ', path)

runName = 'debate-conversations-1-live'
with open('hit-data/{}.json'.format(runName), 'r') as f:
    hits = jsonpickle.decode(f.read(), keys=True, classes=Hit)

hitIndex = orderHitsById(hits)
expertHits = []
for hitId in selectedHitIds:
    expertHits.append(hitIndex[hitId])

saveHits(expertHits, 'debate-expert-sample')