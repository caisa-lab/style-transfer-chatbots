import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

with open('all_blocks.pkl', 'rb') as f:
    allBlocks = pickle.load(f)

aspectResults = pd.read_pickle('AspectSentiment_Results_new.pkl')


def filterAspects():
    if (os.path.isfile('aspects.csv')):
        df = pd.read_csv('aspects.csv', index_col=0)
    else:

        allAspects = defaultdict(int)
        for productId, reviews in aspectResults.items():
            for review in reviews:
                for reviewId, aspect in review:
                    if (aspect):
                        allAspects[aspect.lower()] += 1

        df = pd.DataFrame({
            'aspect': list(allAspects.keys()),
            'occurrence': list(allAspects.values())
        })
        df = df.loc[df['occurrence'] > 20]
        df = df.sort_values(by='occurrence', ascending=False)
        df.to_csv('aspects.csv')
        print(df.head(10))

        sns.histplot(df, y='occurrence')
        plt.show()
    
    # good aspects are marked with a 'x' in the isUseful column
    df = df.loc[pd.notna(df['isUseful'])]
    df.to_csv('aspects_filtered.csv')
    return df

def isBadAspect(aspect, listOfGoodAspects):
    if (not aspect):
        return True
    
    aspect = aspect.lower()
    return aspect not in listOfGoodAspects

df = filterAspects()
goodAspects = df['aspect'].tolist()

# filter all aspect results
for productId, reviews in list(aspectResults.items()):
    for review in reviews:
        badReviews = []
        for key in review:
            reviewId, aspect = key
            if isBadAspect(aspect, goodAspects):
                badReviews.append(key)
        # delete reviews with bad aspects
        for badReview in badReviews:
            del review[badReview]
    
    aspectResults[productId] = [r for r in reviews if len(r) > 0]
    # if there are too few reviews left, remove this product!
    if (len(aspectResults[productId]) <= 10):
        del aspectResults[productId]
    
print('Number of products left after filtering:', len(aspectResults))
pd.to_pickle(aspectResults, 'AspectSentiment_Results_new_filtered.pkl')