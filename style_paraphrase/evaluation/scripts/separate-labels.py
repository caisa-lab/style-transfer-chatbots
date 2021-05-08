import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input_csv, index_col=0)
orgDf = df

for label in df['label'].unique():
    print('label:', label)
    df = orgDf
    df = df.loc[df['label'] == label]
    df.to_csv(args.input_csv.replace('.csv', '.{}.csv'.format(label)))
