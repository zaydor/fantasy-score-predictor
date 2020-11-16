import pandas as pd

# read processed csv files
offense_csv = pd.read_csv('./data/processed/offense.csv')
players_full_csv = pd.read_csv('./data/processed/players_full.csv')

# delete the "unnamed" column
players_full_csv = players_full_csv.dropna(axis=1)

# merge the two processed csv files based on their "player" id
merged = offense_csv.merge(players_full_csv, on='player')

# output merged csv file
merged.to_csv("./data/processed/rb_data.csv", index=False)
