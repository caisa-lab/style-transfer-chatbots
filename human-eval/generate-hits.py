import os
import pandas as pd
from itertools import combinations

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

def getParams(df, style, turnFilter=['bot', 'user'], prefix=''):
    df = df.loc[df['targetStyle'] == style]
    hitParams = {}
    for _, row in df.iterrows():
        turnId = row['turn']
        # skip turns not in filter
        if not any([f in turnId for f in turnFilter]):
            continue
        text = row['text']
        hitParams[prefix + turnId.title()] = text

    return hitParams

def replaceParams(template: str, params: dict) -> str:
    for key, value in params.items():
        # check if value is another dict
        if (hasattr(value, '__keys__')):
            continue
        template = template.replace('%{}%'.format(key), value)
    
    return template

df = pd.read_csv(os.path.join(dname, 'output.csv'), index_col=0)
outputDir = os.path.join(dname, 'hits')
os.makedirs(outputDir, exist_ok=True)

with open(os.path.join(dname, 'templates', 'hit-template.html'), 'r') as f:
    hitTemplate = f.read()

scenario = 'webshop'
baseParams = getParams(df, style='original', turnFilter=['user'])
for style1, style2 in combinations(df['targetStyle'].unique().tolist(), 2):
    turnFilter = ['bot']
    hitParams = dict(baseParams)
    hitParams.update(getParams(df, style1, turnFilter, prefix='style1'))
    hitParams.update(getParams(df, style2, turnFilter, prefix='style2'))
    hitParams.update(style1=style1, style2=style2, scenario=scenario)

    hitFile = replaceParams(hitTemplate, hitParams)

    with open(os.path.join(dname, 'hits', '{}_{}_{}.html'.format(scenario, style1, style2)), 'w') as f:
        f.write(hitFile)


