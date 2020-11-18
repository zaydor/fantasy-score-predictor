import pandas as pd
import argparse
from statistics import mean

parser = argparse.ArgumentParser(description='Preprocess the raw data csv.')
parser.add_argument('--offense', dest='offense_file', action='store', default='./data/raw/offense.csv')
parser.add_argument('--player', dest='players_full_file', action='store', default='./data/raw/players_full.csv')
parser.add_argument('--game', dest='game_file', action='store', default='./data/raw/game.csv')
parser.add_argument('--out', dest='output_file', action='store', default='./data/processed/training_data.csv')
args = parser.parse_args()

years = range(2017,2020)


##### Offense #####
df_offense = pd.read_csv(args.offense_file)

# Delete extraneous columns.
columns_to_drop = ['del', 'first', 'last', 'pa', 'pc', 'py', 'ints', 'tdp', 'ret', 'rety', 'tdret', 'peny', 'conv', 'fp2', 'fp3']
df_offense.drop(columns=columns_to_drop, inplace=True)

# Filter + sort
df_offense = df_offense[df_offense['position1'] == 'RB']
df_offense = df_offense[df_offense['year'].isin(years)]
#df_offense = df_offense.sort_values(['name', 'year', 'week'])


##### Player #####
df_player = pd.read_csv(args.players_full_file)
columns_to_drop = ['first', 'last', 'start', 'cteam', 'dob', 'arm', 'hand', 'dpos', 'col', 'dv', 'jnum', 'dcp']
df_player.drop(columns=columns_to_drop, inplace=True)

# Filter + sort
df_player = df_player[df_player['position1'] == 'RB']
#df_player = df_player.sort_values(['name'])

# Drop the last 2 duplicate columns that were used for filtering and sorting
df_player.drop(columns=['name', 'position1'], inplace=True)


##### Game #####
df_game = pd.read_csv(args.game_file)
df_game = df_game[df_game['seas'].isin(years)]
df_game = df_game[['gid', 'ou']]

##### Merge dataframes #####
df_merged = df_offense.merge(df_player, on='player').dropna(axis=1)
df_merged = df_merged.merge(df_game, on='gid')

df_merged = df_merged.sort_values(['name', 'year', 'week'])

##### Average stats from 3 previous games. #####
df_final = pd.DataFrame(columns=df_merged.columns)
for player in df_merged.name.unique():
    player_rows = df_merged[df_merged['name'] == player]
    for year in player_rows.year.unique():
        year_rows = player_rows[player_rows['year'] == year]
        for i in range(3,len(year_rows)):
            row = year_rows.iloc[i].copy()
            stats_to_average = ['ra', 'sra', 'ry', 'tdr', 'trg', 'rec', 'recy', 'tdrec', 'fuml', 'snp']
            for stat in stats_to_average:
                row[stat] = mean(year_rows[i-3:i][stat])
            df_final = df_final.append(row)

# Move the fantasy points to the last column
cols = list(df_final.columns.values)
cols.pop(cols.index('fp'))
df_final = df_final[cols + ['fp']]

df_final.to_csv(args.output_file, index=False)


