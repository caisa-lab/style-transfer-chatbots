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

scenarioName = 'debate'
OUTPUT_DIR = os.path.join('plots', '{}-conversations'.format(scenarioName))

runName = '{}-conversations-1-live'.format(scenarioName)
df = pd.read_csv('hit-output/{}.assignments.csv'.format(runName), index_col=0)

with open('hit-data/{}.json'.format(runName), 'r') as f:
    hits = jsonpickle.decode(f.read(), keys=True, classes=Hit)
hitIndex = orderHitsById(hits)
df = prepareDataframe(df)
dfBackup = df

#for workerId in ['A1USB9NEEADCTB', 'AX7K5UODLEK72']:
allVsOriginal = []
allRedFlags = []
allMatchups = defaultdict(list)
for agreesWithIssue in df['agreesWithIssue'].unique():
    df = dfBackup
    df = df.loc[df['agreesWithIssue'] == agreesWithIssue]
    fileSuffix = '-agreesWithIssue' + str(agreesWithIssue).title()
    agreesWithChatbot = not agreesWithIssue
    assignments = df

    preferenceResults = getStylePreferences(df).assign(agreesWithChatbot=agreesWithChatbot)

    for matchup in preferenceResults['matchup'].unique():
        currentResults = preferenceResults.loc[preferenceResults['matchup'] == matchup]
        allMatchups[matchup].append(currentResults)
        sns.catplot(x='votePercent', y='question', hue='style', kind='bar', data=currentResults)
        plt.title('All styles vs original')
        plt.xlabel('Percentage of votes')
        plt.ylabel('Question')
        plt.title('Matchup: ' + matchup)
        saveFigure(os.path.join(OUTPUT_DIR, 'preferences', '{}{}.png'.format(matchup, fileSuffix)))

    # plot all styles vs original
    df = preferenceResults
    df = df.loc[df['matchup'].str.contains('original')]
    # do not show original, since it is basically 1 - otherStyle
    df = df.loc[df['style'] != 'original'].sort_values(by='style', key=sortStyleSeries)

    allVsOriginal.append(df)
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
    allRedFlags.append(redFlagResults.assign(agreesWithChatbot=agreesWithChatbot))
    redFlagOrder = ['Offensive', 'Incoherent']
    sns.catplot(x='meanFlagPercent', y='flag', hue='style', kind='bar', data=redFlagResults, order=redFlagOrder)
    #plt.title('Percentage of flagged messages per conversation for each style')
    plt.xlabel('Mean percentage of flagged messages per conversation')
    plt.ylabel('Flag')
    saveFigure(os.path.join(OUTPUT_DIR, 'red-flags{}.png'.format(fileSuffix)))

allVsOriginal = pd.concat(allVsOriginal, ignore_index=True)
allRedFlags = pd.concat(allRedFlags, ignore_index=True)

g = sns.catplot(x='votePercent', y='question', hue='style', kind='bar', col='agreesWithChatbot', data=allVsOriginal)
g.axes[0, 0].set_ylabel('Question')
for axis in g.axes[0]:
    axis.set_xlabel('Percentage of votes')
saveFigure(os.path.join(OUTPUT_DIR, 'receptive', 'all vs original.png'))

redFlagOrder = ['Offensive', 'Incoherent']
g = sns.catplot(x='meanFlagPercent', y='flag', hue='style', kind='bar', col='agreesWithChatbot', data=allRedFlags, order=redFlagOrder)
g.axes[0, 0].set_ylabel('Flag')
for axis in g.axes[0]:
    axis.set_xlabel('Mean percentage of flagged messages per conversation')
saveFigure(os.path.join(OUTPUT_DIR, 'receptive', 'red-flags.png'))

for matchup in allMatchups.keys():
    receptiveMatchup = pd.concat(allMatchups[matchup], ignore_index=True)
    g = sns.catplot(x='votePercent', y='question', hue='style', kind='bar', col='agreesWithChatbot', data=receptiveMatchup)
    g.axes[0, 0].set_ylabel('Question')
    for axis in g.axes[0]:
        axis.set_xlabel('Percentage of votes')

    saveFigure(os.path.join(OUTPUT_DIR, 'receptive', '{}.png'.format(matchup)))