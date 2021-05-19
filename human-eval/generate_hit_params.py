import os
import json
import random
import copy
import pandas as pd
from itertools import combinations
from collections import defaultdict

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

def updateParams(df, style, hitParams, turnFilter=['bot', 'user'], prefix=''):
    df = df.loc[df['targetStyle'] == style]
    df = df.loc[df['sender'].isin(turnFilter)]
    for _, row in df.iterrows():
        turnId = row['turnId']
        text = row['text']
        sender = row['sender']
        isConst = row['isConst']
        hitParams['turns'][turnId]['{}{}Message'.format(sender, prefix)] = text
        hitParams['turns'][turnId]['sender'] = sender
        hitParams['turns'][turnId]['isConst'] = isConst

    return hitParams

def isFromSameDomain(styles):
    if ('original' in styles):
        return True
    
    style1, style2 = sorted(styles, key=len)
    return style1.lower() in style2.lower()
    
def generateHitParamsStyle(df):
    with open(os.path.join(dname, 'templates', 'scenarios.json'), 'r') as f:
        scenarios = json.load(f)

    dfBackup = df
    allParams = []
    for conversationId in dfBackup['conversationId'].unique():
        df = dfBackup
        df = df.loc[df['conversationId'] == conversationId]
        scenario = df['scenario'].iloc[0]

        baseParams = {}
        baseParams['turns'] = defaultdict(dict)
        baseParams = updateParams(df, style='original', hitParams=baseParams, turnFilter=['user'])

        for styles in combinations(df['targetStyle'].unique().tolist(), 2):
            styles = list(styles)
            if (not isFromSameDomain(styles)):
                continue
            # shuffle in-place
            random.shuffle(styles)
            style1, style2 = styles
            turnFilter = ['bot']

            hitParams = copy.deepcopy(baseParams)
            hitParams = updateParams(df, style1, hitParams, turnFilter, prefix='1')
            hitParams = updateParams(df, style2, hitParams, turnFilter, prefix='2')
            hitParams = dict(hitParams)
            hitParams.update({
                'style1': style1,
                'style2': style2,
                'scenario': scenario,
                'conversationId': int(conversationId)
            })
            hitParams.update(scenarios[scenario])
            allParams.append(hitParams)

    return allParams

def generateHitParamsCoherence(df):
    with open(os.path.join(dname, 'templates', 'scenarios.json'), 'r') as f:
        scenarios = json.load(f)
    
    dfBackup = df
    allParams = []
    for conversationId in dfBackup['conversationId'].unique():
        df = dfBackup
        df = df.loc[df['conversationId'] == conversationId]
        scenario = df['scenario'].iloc[0]
        baseParams = {}
        baseParams['turns'] = defaultdict(dict)
        baseParams = updateParams(df, style='original', hitParams=baseParams, turnFilter=['user'])

        turnFilter = ['bot']
        hitParams = copy.deepcopy(baseParams)
        hitParams = updateParams(df, 'original', hitParams, turnFilter)
        hitParams = dict(hitParams)
        hitParams.update({
            'scenario': scenario,
            'conversationId': int(conversationId)
        })
        hitParams.update(scenarios[scenario])
        allParams.append(hitParams)

    return allParams