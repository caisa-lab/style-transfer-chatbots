import pandas as pd
from sqlalchemy import create_engine
import os
import ast

def sqlToCsv():
    engine = create_engine('postgresql://philno@localhost:5432/dasp')
    table = 'users'
    df = pd.read_sql_table(table, engine)

    df.to_csv('dasp/{}.csv'.format(table))

DASP_PATH = '/data/daten/datasets/dasp/dasp_csv'
df = pd.read_csv(os.path.join(DASP_PATH, 'posts.csv'))
df = df.loc[pd.notna(df['topic'])]
df = df.loc[df['origin'] == 'REDDIT']
df = df.loc[df['content'] != '[removed]']
df = df.loc[df['content'] != '[deleted]']
df = df.loc[pd.notna(df['content'])]

contentLengths = []
scores = []
controversials = []
for index, row in df.iterrows():
    contentLengths.append(len(row['content']))

    metaData = ast.literal_eval(row['platform_specific'])
    scores.append(metaData['score'])
    if ('controversiality' in metaData):
        controversials.append(metaData['controversiality'])
    else:
        controversials.append(0)

df = df.assign(contentLength=contentLengths, score=scores, controversial=controversials)
df.to_csv(os.path.join(DASP_PATH, 'posts_reddit_topic.csv'))
df = df.filter(['content', 'sentiment', 'stance', 'topic', 'url', 'contentLength', 'score', 'controversial'])
controversial = df.loc[df['controversial'] > 0]
controversial.to_csv(os.path.join(DASP_PATH, 'posts_reddit_filtered_controversial.csv'))
highScore = df.loc[df['score'] >= 4]
highScore.to_csv(os.path.join(DASP_PATH, 'posts_reddit_filtered_high_score.csv'))
print('yolo123')