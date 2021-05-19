import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from inflection import titleize
from pathlib import Path
from my_plot_utils import saveFigure, prepareDataframe, getStylePreferences, sortStyleSeries, getPreferenceQuestionOrder

sns.set_theme('notebook', 'darkgrid', palette='colorblind')

demo = pd.read_csv('hit-output/demographics-live.csv', index_col=0)

batch1 = pd.read_csv('hit-output/phone-conversations-1-live.workers.csv', index_col=0)
batch2 = pd.read_csv('hit-output/debate-conversations-1-live.workers.csv', index_col=0)

workerDict = defaultdict(int)
minAssignments = 5

for df in [batch1, batch2]:
    for idx, row in df.iterrows():
        workerId = row['workerId']
        numOfAssignments = row['numOfAssignments']
        if (numOfAssignments < minAssignments):
            continue
        #else
        workerDict[str(workerId)] += numOfAssignments

participants = []
for workerId, numOfAssignments in workerDict.items():
    # worker whitelist
    #if (workerId not in [
    #    'A1USB9NEEADCTB',
    #    'AX7K5UODLEK72'
    #]):
    #    continue

    if (numOfAssignments < 2 * minAssignments):
        continue
    # else
    participants.append({
        'workerId': workerId,
        'numOfAssignments': numOfAssignments
    })

participants = pd.DataFrame(participants)
print('sample size:', len(participants))
participants.hist('numOfAssignments')
#plt.show()
plt.clf()

demo = demo.loc[demo['WorkerId'].isin(participants['workerId'])]

def plotDemographics(demo):
    plotDf = []
    for column in 'age,country,education,gender'.split(','):
        counts = demo[column].value_counts()
        for value, count in counts.iteritems():
            value = value.replace('_', ' to ')
            if (value == 'usa'):
                value = 'USA'
            else:
                value = titleize(value)
            
            plotDf.append({
                'question': titleize(column),
                'answer': value,
                'count': count
            })

    plotDf = pd.DataFrame(plotDf)
    order = []
    for question in plotDf['question'].unique():
        df = plotDf
        df = df.loc[df['question'] == question]
        if (question == 'Education'):
            order += ['Highschool', 'Bachelor', 'Master']
        else:
            order += list(sorted(df['answer'].unique().tolist()))

    g = sns.barplot(data=plotDf, y='answer', x='count', hue='question', dodge=False, order=order)
    #sns.countplot(data=plotDf, y='answer', x='count', hue='question')
    plt.xlabel('Number of workers')
    plt.ylabel('Answer')
    plt.legend(title='Question')
    return plotDf

plotDemographics(demo)
saveFigure('plots/demographics-live.png')

assignments1 = pd.read_csv('hit-output/phone-conversations-1-live.assignments.csv', index_col=0)
assignments2 = pd.read_csv('hit-output/debate-conversations-1-live.assignments.csv', index_col=0)
assignments = pd.concat([assignments1, assignments2], ignore_index=True)
#assignments = assignments2

# no errors, no captchas
df = prepareDataframe(assignments)
# only workers with min. number of assignments
df = df.loc[df['workerId'].isin(participants['workerId'])]

workerNameMap = {
    'A1USB9NEEADCTB': 'Adam',
    'AX7K5UODLEK72': 'Bob'
}

orgDf = df
workerPrefs = []
workerDemos = []
for scenario in orgDf['scenario'].unique().tolist():
    for workerId in participants['workerId']:
        df = orgDf
        df = df.loc[df['scenario'] == scenario]
        df = df.loc[df['workerId'] == workerId]

        currentPrefs = getStylePreferences(df, minSupport=10)
        currentPrefs = currentPrefs.assign(scenario=scenario)
        if (len(currentPrefs) < 1):
            continue
        workerDemo = demo.loc[demo['WorkerId'] == workerId]
        filterColumns = 'age,country,education,gender'.split(',')
        workerDemo = workerDemo[filterColumns]
        workerDemos.append(workerDemo)
        currentPrefs = currentPrefs.assign(**workerDemo.iloc[0].to_dict())
        currentPrefs = currentPrefs.assign(workerId=workerId)
        #currentPrefs = currentPrefs.assign(worker=workerNameMap[workerId])
        workerPrefs.append(currentPrefs)

print('num of workers:', len(workerPrefs))
workerPrefs = pd.concat(workerPrefs, ignore_index=True)
workerDemos = pd.concat(workerDemos, ignore_index=True)

plotDemographics(workerDemos)
plt.show()

df = workerPrefs
df = df.loc[df['matchup'].str.contains('original')]
# do not show original, since it is basically 1 - otherStyle
df = df.loc[df['style'] != 'original'].sort_values(by='style', key=sortStyleSeries)


# regular plot:
# sns.catplot(x='votePercent', y='question', hue='style', kind='bar', data=df)

#g = sns.catplot(x='votePercent', y='question', hue='style', col='age', kind='box', data=df)
g = sns.catplot(x='votePercent', y='question', hue='style', kind='box', col='scenario', data=df, order=getPreferenceQuestionOrder())
g.axes[0, 0].set_ylabel('Question')
lowestAxis = len(g.axes) - 1
g.axes[lowestAxis, 0].set_ylabel('Question')
for axis in g.axes[lowestAxis]:
    axis.set_xlabel('Percentage of votes')

#plt.show()
saveFigure('plots/personal-preferences-all-participants.png')
#saveFigure('plots/demo-age-all-vs-original.png')