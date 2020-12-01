import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Preprocess the raw data csv.')
parser.add_argument('--file', dest='offense_file', action='store', required=True)
parser.add_argument('--out', dest='output_file', action='store', default='./data/processed/offense.csv')
args = parser.parse_args()

df = pd.read_csv(args.offense_file)

# Delete extraneous columns.
columns_to_drop = ['uid', 'gid', 'del', 'first', 'last',
                   'pa', 'pc', 'py', 'ints', 'tdp', 'ret',
                   'rety', 'tdret', 'peny', 'conv']
df.drop(columns=columns_to_drop, inplace=True)

# Filter + sort
df = df[df['position1'] == 'RB']
df = df[df['year'].isin([2017, 2018, 2019])]
df = df.sort_values(['name', 'year', 'week'])

df.to_csv(args.output_file, index=False)

