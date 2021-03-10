import os
import json
import random
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
        hitParams['turns'][turnId]['{}{}Message'.format(sender, prefix)] = text
        hitParams['turns'][turnId]['sender'] = sender

    return hitParams

def replaceParams(template: str, params: dict) -> str:
    for key, value in params.items():
        # check if value is another dict
        if (hasattr(value, 'keys')):
            continue
        template = template.replace('%{}%'.format(key), value)
    
    return template

def getTurn(turnTemplate: str, params: dict) -> str:
    tempParams = {
        'userMessage': '',
        'bot1Message': '',
        'bot2Message': ''
    }
    tempParams.update(params)
    return replaceParams(turnTemplate, tempParams)

def isFromSameDomain(styles):
    if ('original' in styles):
        return True
    
    style1, style2 = sorted(styles, key=len)
    return style1.lower() in style2.lower()
    

df = pd.read_csv(os.path.join(dname, 'output.csv'), index_col=0)
outputDir = os.path.join(dname, 'hits')
os.makedirs(outputDir, exist_ok=True)

with open(os.path.join(dname, 'templates', 'hit-template.html'), 'r') as f:
    hitTemplate = f.read()
with open(os.path.join(dname, 'templates', 'turn.html'), 'r') as f:
    turnTemplate = f.read()
with open(os.path.join(dname, 'templates', 'scenarios.json'), 'r') as f:
    scenarios = json.load(f)

dfBackup = df
for scenario in dfBackup['scenario'].unique():
    df = dfBackup
    df = df.loc[df['scenario'] == scenario]
    baseParams = defaultdict(lambda: defaultdict(dict))
    updateParams(df, style='original', hitParams=baseParams, turnFilter=['user'])

    for styles in combinations(df['targetStyle'].unique().tolist(), 2):
        styles = list(styles)
        if (not isFromSameDomain(styles)):
            continue
        # shuffle in-place
        random.shuffle(styles)
        style1, style2 = styles
        turnFilter = ['bot']

        hitParams = defaultdict(lambda: defaultdict(dict))
        hitParams.update(dict(baseParams))
        updateParams(df, style1, hitParams, turnFilter, prefix='1')
        updateParams(df, style2, hitParams, turnFilter, prefix='2')
        hitParams.update({
            'style1': style1,
            'style2': style2,
            'scenario': scenario
        })

        hitFile = replaceParams(hitTemplate, hitParams)

        turnsHtml = ''
        for turn, params in sorted(hitParams['turns'].items(), key=lambda x: x[0]):
            turnsHtml += getTurn(turnTemplate, params)
            turnsHtml += '\n'

        hitFile = replaceParams(hitFile, { 'turns': turnsHtml })
        hitFile = replaceParams(hitFile, scenarios[scenario])

        with open(os.path.join(dname, 'hits', '{}_{}_{}.html'.format(scenario, style1, style2)), 'w') as f:
            f.write(hitFile)


