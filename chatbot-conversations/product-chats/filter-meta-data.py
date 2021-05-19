import pandas as pd

df = pd.read_csv('cell_phones_df.csv', index_col=0)

columnSubset = 'category_0,category_1,category_2,description,title,image,brand,rank,main_cat,date,price,num_ratings,avg_rating,asin'.split(',')
df = df[columnSubset]
df.to_csv('cell_phones_df_filtered.csv')

retrievedItemsDict = pd.read_pickle('retrieved_items_dict.pkl')
knownItems = set()
phoneFeatures = {}

for entry in retrievedItemsDict.values():
    for itemId in entry['retrieved items']:
        if (itemId in knownItems):
            continue
        else:
            phoneFeatures[itemId] = entry['preferences']
            knownItems.add(itemId)

pd.to_pickle(phoneFeatures, 'phoneFeatures.pkl')