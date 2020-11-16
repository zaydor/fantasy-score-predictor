import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Preprocess the raw player attribute csv.')
parser.add_argument('--file', dest='players_full_file', action='store', required=True)
parser.add_argument('--out', dest='output_file', action='store', default='./data/processed/players_full.csv')
args = parser.parse_args()

df = pd.read_csv(args.players_full_file)

# Delete extraneous columns (including columns that are already in the processed offense.csv).
columns_to_drop = ['first', 'last', 'start', 'cteam', 'dob', 'arm', 'hand', 'dpos', 'col', 'dv', 'jnum', 'dcp']

df.drop(columns=columns_to_drop, inplace=True)

# Filter + sort
df = df[df['position1'] == 'RB']
df = df.sort_values(['name'])

# Drop the last 2 duplicate columns that were used for filtering and sorting
df.drop(columns=['name', 'position1'], inplace=True)

df.to_csv(args.output_file, index=False)
