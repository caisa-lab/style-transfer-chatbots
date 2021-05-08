from os import pardir
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True)
parser.add_argument('--output_txt', type=str, required=True)
parser.add_argument('--column_name', type=str, required=True)

args = parser.parse_args()

print('Input:', args.input_csv)
print('Extracting column:', args.column_name)
print('Output:', args.output_txt)


df = pd.read_csv(args.input_csv, index_col=0)
values = df[args.column_name].tolist()

with open(args.output_txt, 'w') as f:
    f.write('\n'.join(values))