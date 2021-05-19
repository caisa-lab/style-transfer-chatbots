from collections import defaultdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import jsonpickle
from inflection import titleize
from hit import Hit
from hit_list import orderHitsById

from my_plot_utils import saveFigure, prepareDataframe, getStylePreferences, sortStyleSeries

scenarioName = 'phone'
OUTPUT_DIR = os.path.join('plots', '{}-conversations'.format(scenarioName))

runName = '{}-conversations-1-live'.format(scenarioName)
df = pd.read_csv('hit-output/{}.assignments.csv'.format(runName), index_col=0)

with open('hit-data/{}.json'.format(runName), 'r') as f:
    hits = jsonpickle.decode(f.read(), keys=True, classes=Hit)
hitIndex = orderHitsById(hits)
df = prepareDataframe(df)
dfBackup = df

#for workerId in ['A1USB9NEEADCTB', 'AX7K5UODLEK72']:
#for agreesWithIssue in df['agreesWithIssue'].unique():
df = dfBackup
#df = df.loc[df['agreesWithIssue'] == agreesWithIssue]
#fileSuffix = '-agreesWithIssue' + str(agreesWithIssue).title()
fileSuffix = ''
assignments = df

preferenceResults = getStylePreferences(df)
prefTable = []
for matchup in preferenceResults['matchup'].unique():
    currentResults = preferenceResults.loc[preferenceResults['matchup'] == matchup]
    sns.catplot(x='votePercent', y='question', hue='style', kind='bar', data=currentResults)
    plt.xlabel('Percentage of votes')
    plt.ylabel('Question')
    plt.title('Matchup: ' + matchup)
    saveFigure(os.path.join(OUTPUT_DIR, 'preferences', '{}{}.png'.format(matchup, fileSuffix)))

    if ('original' in matchup.lower()):
        continue
    currentRow = {
        'Style Pair': titleize(matchup),   
    }
    for question in currentResults['question'].unique().tolist():
        df = currentResults
        df = df.loc[df['question'] == question].sort_values(by='votePercent', ascending=False)
        best = df.iloc[0]
        bestName = titleize(best['style'])
        bestValue = round(best['votePercent'] * 100, 1) 
        questionName = titleize(question)
        currentRow[questionName] = '{} ({}%)'.format(bestName, bestValue)
    prefTable.append(currentRow)

pd.DataFrame(prefTable).to_csv(os.path.join(OUTPUT_DIR, 'preferences', 'pref-table.csv'))
df = preferenceResults

# get general style preferences

df.to_csv(os.path.join(OUTPUT_DIR, 'preferences', 'preference-results.csv'))
# plot all styles vs original
df = df.loc[df['matchup'].str.contains('original')]
# do not show original, since it is basically 1 - otherStyle
df = df.loc[df['style'] != 'original'].sort_values(by='style', key=sortStyleSeries)
sns.catplot(x='votePercent', y='question', hue='style', kind='bar', data=df)
plt.title('All styles vs. original')
plt.xlabel('Percentage of votes')
plt.ylabel('Question')
saveFigure(os.path.join(OUTPUT_DIR, 'preferences', 'all vs original{}.png'.format(fileSuffix)))

redFlags = []

for idx, row in assignments.iterrows():
    hit = hitIndex[row['hitId']]
    turnCount = len(hit.getBotTurns(isConst=False))
    for styleIndex in [1, 2]:
        styleName = row['style{}'.format(styleIndex)]
        for flag in ['Offensive[]', 'Incoherent[]']:
            flagColumn = 'bot{}{}'.format(styleIndex, flag)

            flagValue = row[flagColumn]
            redFlagCount = 0
            if (pd.notna(flagValue) and len(flagValue) > 0):
                redFlagCount = len(flagValue.split('|'))
            
            redFlagPercent = float(redFlagCount) / float(turnCount)

            redFlags.append({
                'style': styleName,
                'flagPercent': redFlagPercent,
                'flag': flag.replace('[]', '')
            })

redFlags = pd.DataFrame(redFlags).sort_values(by='style', key=sortStyleSeries)
redFlagResults = []
for style in redFlags['style'].unique():
    for flag in redFlags['flag'].unique():
        df = redFlags
        df = df.loc[df['style'] == style]
        df = df.loc[df['flag'] == flag]
        redFlagResults.append({
            'style': style,
            'flag': flag,
            'meanFlagPercent': df['flagPercent'].mean()
        })

redFlagResults = pd.DataFrame(redFlagResults).sort_values(by='style', key=sortStyleSeries)
redFlagOrder = ['Offensive', 'Incoherent']
sns.catplot(x='meanFlagPercent', y='flag', hue='style', kind='bar', data=redFlagResults, order=redFlagOrder)
#plt.title('Percentage of flagged messages per conversation for each style')
plt.xlabel('Mean percentage of flagged messages per conversation')
plt.ylabel('Flag')
saveFigure(os.path.join(OUTPUT_DIR, 'red-flags{}.png'.format(fileSuffix)))


# plot correlation

# convert the styles to numerical labels
preferenceColumns = 'moreNatural,moreLikeable,personalPreference'.split(',')
numericalLabels = defaultdict(list)

df = assignments
for idx, row in df.iterrows():
    style1 = row['style1'].lower()
    style2 = row['style2'].lower()
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
    
    moreOffensive = 1
    moreIncoherent = 1
    incoherentScore = 0
    offensiveScore = 0
    for styleIndex in [1, 2]:
        for flag in ['Offensive[]', 'Incoherent[]']:
            flagColumn = 'bot{}{}'.format(styleIndex, flag)

            flagValue = row[flagColumn]
            redFlagCount = 0
            if (pd.notna(flagValue) and len(flagValue) > 0):
                redFlagCount = len(flagValue.split('|'))
            
            if ('offensive' in flag.lower()):
                if (redFlagCount > offensiveScore):
                    moreOffensive = 0 if (styleIndex == 1) else 2
                    offensiveScore = redFlagCount
            else:
                if (redFlagCount > incoherentScore):
                    moreIncoherent = 0 if (styleIndex == 1) else 2
                    incoherentScore = redFlagCount
    
    numericalLabels['More Offensive'].append(moreOffensive)
    numericalLabels['More Incoherent'].append(moreIncoherent)

df = df.assign(**numericalLabels)

plotLabels = [titleize(l.replace('Num', '')) for l in numericalLabels.keys()]
for method in ['pearson', 'spearman', 'kendall']:
    corr = df[list(numericalLabels.keys())].corr(method=method)
    sns.heatmap(corr, vmin=-1, vmax=1, center=0, annot=True, xticklabels=plotLabels, yticklabels=plotLabels)
    #plt.title('{} correlation of questions and flags'.format(method.title()))
    plt.yticks(rotation=0)
    saveFigure(os.path.join(OUTPUT_DIR, 'correlation', '{}{}.png'.format(method, fileSuffix)))