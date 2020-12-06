import pandas as pd
import argparse
from statistics import mean

def get_previous_week_and_year(week, year):
    if week > 1:
        week = week - 1
    else:
        week = 17
        year -= 1
    return week, year

def get_opponent_dst_stats_for_previous_weeks(opponent, week, year, df_dst):
    ''' Collect the 3 previous weeks of opponent's dst performances.'''
    weeks = [(week + i) % 17 + 1 for i in range(12,16)] # grab the 4 previous weeks, in case of bye
    years = [year] * 4

    # check if we wrapped to the previous year
    for i in range(4):
        if weeks[i] > week:
            years[i] -= 1

    # we have the previous weeks/years, now look them up and average them
    ypa = [] # yards allowed per attempt
    rtd = [] # rushing td's allowed
    for i in range(4):
        try:
            df_game = df_dst[(df_dst['team'] == opponent) & (df_dst['week'] == weeks[i]) & (df_dst['year'] == years[i])]
            ypa.append(df_game.iloc[0]['Y/A'])
            rtd.append(df_game.iloc[0]['TD'])
        except:
            # probably a bye week
            pass

    return mean(ypa[-3:]), mean(rtd[-3:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the raw data csv.')
    parser.add_argument('--offense', dest='offense_file', action='store', default='./data/raw/offense.csv')
    parser.add_argument('--player', dest='players_full_file', action='store', default='./data/raw/players_full.csv')
    parser.add_argument('--game', dest='game_file', action='store', default='./data/raw/game.csv')
    parser.add_argument('--dst', dest='dst_file', action='store', default='./data/raw/dst_stats.csv')
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
    df_offense = df_offense[df_offense['week'] <= 17]
    #df_offense = df_offense.sort_values(['name', 'year', 'week'])


    ##### Player #####
    df_player = pd.read_csv(args.players_full_file)
    columns_to_drop = ['first', 'last', 'start', 'cteam', 'dob', 'arm', 'hand', 'dpos', 'col', 'dv', 'jnum', 'dcp']
    df_player.drop(columns=columns_to_drop, inplace=True)
    df_player = df_player.replace({'forty': 0}, 4.49)
    df_player = df_player.replace({'bench': 0}, 20.43)
    df_player = df_player.replace({'vertical': 0}, 35.1)
    df_player = df_player.replace({'broad': 0}, 118.87)
    df_player = df_player.replace({'shuttle': 0}, 4.28)
    df_player = df_player.replace({'cone': 0}, 6.99)

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

    ##### Retrieve opponent dst data. #####
    df_dst = pd.read_csv(args.dst_file)
    df_final['d_ypa'] = 4.2 # this is the NFL average
    df_final['d_rtd'] = 0.88 # the 2019 average
    for i, row in df_final.iterrows():
        # attributes for this game
        team, week, year = row['team'], row['week'], row['year']

        # lookup the opponent
        df_off_game = df_dst[(df_dst['team'] == team) & (df_dst['week'] == week) & (df_dst['year'] == year)]
        opponent = df_off_game.iloc[0]['Opp']

        ypa, rtd = get_opponent_dst_stats_for_previous_weeks(opponent, week, year, df_dst)
        df_final.at[i, 'd_ypa'] = ypa
        df_final.at[i, 'd_rtd'] = rtd

    # Move the fantasy points to the last column
    cols = list(df_final.columns.values)
    cols.pop(cols.index('fp'))
    df_final = df_final[cols + ['fp']]

    df_final.to_csv(args.output_file, index=False)
