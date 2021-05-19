import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from inflection import titleize

sns.set_theme('notebook', 'darkgrid', palette='colorblind')

allStyles = ['formal', 'informal', 'polite', 'impolite', 'receptive', 'not_receptive', 'original']

styleSortOrder = {}
for idx, style in enumerate(allStyles):
    styleSortOrder[style] = idx

def saveFigure(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=800)
    plt.clf()

def sortStyle(styleName: str):
    return styleSortOrder[styleName]

def sortStyleSeries(styleSeries):
    return styleSeries.map(sortStyle)

def sortMatchups(matchups):    
    result = pd.Series(dtype='int')
    for idx, matchup in matchups.iteritems():
        styles = matchup.split(', ')
        result.loc[idx] = sortStyle(styles[0]) * 10 + sortStyle(styles[1])
    return result

def prepareDataframe(df):
    df = df.loc[df['hitType'] == 'task']
    df = df.loc[df['error'] != True]
    df = df.loc[pd.isna(df['error'])]

    matchups = []
    for idx, row in df.iterrows():
        matchup = sorted([row['style1'], row['style2']], key=sortStyle)
        matchups.append(', '.join(matchup))

    df = df.assign(styleMatchup=matchups)
    df = df.sort_values(by='styleMatchup', key=sortMatchups)
    return df

def getStylePreferences(df, minSupport=0):
    preferenceResults = pd.DataFrame()
    orgDf = df
    for matchup in orgDf['styleMatchup'].unique():
        currentResults = []
        df = orgDf
        df = df.loc[df['styleMatchup'] == matchup]

        # skip if only very few instances were annotated
        if (len(df) < minSupport):
            continue

        for key in 'moreNatural,moreLikeable,personalPreference'.split(','):
            valueCounts = df[key].value_counts(normalize=True)
            for style, count in valueCounts.iteritems():
                currentResults.append({
                    'matchup': matchup,
                    'question': titleize(key),
                    'style': style,
                    'votePercent': count 
                })
        currentResults = pd.DataFrame(currentResults).sort_values(by='style', key=sortStyleSeries)
        preferenceResults = pd.concat([preferenceResults, currentResults], ignore_index=True)
    
    return preferenceResults

def getPreferenceQuestionOrder():
    return ['More Natural', 'More Likeable', 'Personal Preference']