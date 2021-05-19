import simpledorff
from disagree import metrics
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import jsonpickle
from hit import Hit
from hit_list import orderHitsById

#runName = 'phone-conversations-1-live'
#runName = 'caisa-test-1'
runName = 'debate-expert-hits-sandbox'

def getAssignmentsForRun(runName):
    df = pd.read_csv('hit-output/{}.assignments.csv'.format(runName), index_col=0)
    df = df.loc[df['hitType'] == 'task']
    df = df.loc[df['error'] != True]
    df = df.loc[pd.isna(df['error'])]
    return df

workerDf = getAssignmentsForRun('debate-conversations-1-live')
expertDf = getAssignmentsForRun(runName)

with open('hit-data/{}.json'.format(runName), 'r') as f:
    hits = jsonpickle.decode(f.read(), keys=True, classes=Hit)
hitIndex = orderHitsById(hits)

newHitIds = []
for idx, row in expertDf.iterrows():
    hit = hitIndex[row['hitId']]

    newHitIds.append(hit.parameters['workerHitId'])

df = expertDf
df = df.drop(columns=['hitId']).assign(hitId=newHitIds)
workerDf = workerDf.loc[workerDf['hitId'].isin(df['hitId'].unique())]
df = pd.concat([df, workerDf], ignore_index=True)

# convert the styles to numerical labels
preferenceColumns = 'moreNatural,moreLikeable,personalPreference'.split(',')
#preferenceColumns = 'moreNatural,moreEngaging,personalPreference'.split(',')
numericalLabels = defaultdict(list)

combinedLabelsDict = {}
labelCtr = 0
for a in range(2):
    for b in range(2):
        for c in range(2):
            labelStr = '{}{}{}'.format(a, b, c)
            combinedLabelsDict[labelStr] = labelCtr
            labelCtr += 1


for idx, row in df.iterrows():
    style1 = row['style1'].lower()
    style2 = row['style2'].lower()
    combinedPreferences = []
    for prefColumn in preferenceColumns:
        style = row[prefColumn].lower()
        numericalLabel = None
        if (style == style1):
            numericalLabel = 0
        elif (style == style2):
            numericalLabel = 1
        else:
            print('style not found!')
        
        numericalLabels[prefColumn + 'Num'].append(numericalLabel)
        combinedPreferences.append(str(numericalLabel))
    
    numericalLabels['preferenceCombined'].append(combinedLabelsDict[''.join(combinedPreferences)])

def testRandomAnswers():
    for key in dict(numericalLabels):
        labelRange = (0,2)
        if (key == 'preferenceCombined'):
            labelRange = (0,8)
        numericalLabels[key] = np.random.randint(*labelRange, size=len(numericalLabels[key]))

df = df.assign(**numericalLabels)

print('simpledorff:')
for numericalLabel in numericalLabels.keys():
    alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
        df,
        experiment_col='hitId',
        annotator_col='workerId',
        class_col=numericalLabel
    )
    print('Krippendorffs alpha for {}:'.format(numericalLabel.replace('Num', '')), alpha)

print('disagree:')
MACE_TEMP_DIR = 'temp/{}'.format(runName)
os.makedirs(MACE_TEMP_DIR, exist_ok=True)
workerIds = df['workerId'].unique()
backupDf = df
for numericalLabel in numericalLabels.keys():
    df = backupDf
    workerDict = defaultdict(list)
    for hitId in df['hitId'].unique():
        for workerId in workerIds:
            df = backupDf
            df = df.loc[df['hitId'] == hitId]
            df = df.loc[df['workerId'] == workerId]
            if (len(df) > 0):
                workerDict[workerId].append(df[numericalLabel].item())
            else:
                workerDict[workerId].append(None)

    workerDict = pd.DataFrame(workerDict)
    workerDict.to_csv('{}/mace-table-{}.csv'.format(MACE_TEMP_DIR, numericalLabel), header=False, index=False)
    pd.to_pickle(workerDict.columns.tolist(), '{}/worker-ids-{}.pkl'.format(MACE_TEMP_DIR, numericalLabel))

    kripp = metrics.Krippendorff(workerDict, True)
    alpha = kripp.alpha(data_type='nominal')
    print('Krippendorffs alpha for {}:'.format(numericalLabel.replace('Num', '')), alpha)