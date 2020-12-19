"""
Authors: Isaiah Thomas and Dillon Johnson
File: 
Date: 12/18/2020
Class: CS596

Helper methods to get consistent training, validation, and test data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

class FantasyFootballData:
    """Helper class to filter and split training and test data."""

    default_csv = 'data/processed/training_data.csv'
    default_features = ['sra','tdr','trg','tdrec','fuml','snp','seas', \
                'weight','forty','bench','broad','shuttle','cone', \
                'ou','d_ypa','d_rtd']

    all_features = ['ra','sra','ry','tdr','trg','rec','recy','tdrec','fuml','snp','seas', \
        'height','weight','forty','bench','vertical','broad','shuttle','cone', \
        'ou','d_ypa','d_rtd']

    def __init__(self, csv=default_csv, features=default_features):
        # Read in file
        self._csv = csv
        df = pd.read_csv(csv)

        # Just get rid of any rows with blank cells.
        df.dropna(inplace=True)
        df = df.sample(frac=1)

        test = df[(df['year'] == 2019) & (df['week'] == 16)]
        train = df[(df['year'] != 2019) | (df['week'] != 16)]

        evaluation = train[(train['year'] == 2018) & (df['week'] == 16)]
        train = train[(train['year'] != 2018) | (train['week'] != 16)]

        # Split up and preprocess the data.
        self.y_train = train['fp']
        x_train = self._handle_prescale_actions(train[features])
        scaler = StandardScaler()
        scaler.fit(x_train)
        self.x_train = scaler.transform(x_train)
        self.train_player_performances = train[['name', 'fp']]

        self.y_eval = evaluation['fp']
        x_eval = self._handle_prescale_actions(evaluation[features])
        scaler = StandardScaler()
        scaler.fit(x_eval)
        self.x_eval = scaler.transform(x_eval)
        self.eval_player_performances = evaluation[['name','fp']]

        self.y_test = test['fp']
        x_test = self._handle_prescale_actions(test[features])
        scaler = StandardScaler()
        scaler.fit(x_test)
        self.x_test = scaler.transform(x_test)
        self.test_player_performances = test[['name', 'fp']]

    def _handle_prescale_actions(self, x):
        return x

    def get_training_data(self, random_state=None):
        return self.x_train, self.x_eval, self.y_train, self.y_eval

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_train_player_performances(self):
        return self.train_player_performances

    def get_eval_player_performances(self):
        return self.eval_player_performances

    def get_test_player_performances(self):
        return self.test_player_performances

class PolyFantasyFootballData(FantasyFootballData):
    """Extend FantasyFootballData to include polynomial features."""

    def __init__(self, csv=FantasyFootballData.default_csv, features=FantasyFootballData.default_features, degree=2):
        self.degree = degree
        super().__init__(csv=csv, features=features)

    def _handle_prescale_actions(self, x):
        poly = PolynomialFeatures(degree=self.degree)
        return poly.fit_transform(x)

class FFEvaluation:

    def __init__(self, results):
        self.results = results

    def get_rankings(self):
        results = self.results.sort_values('fp', ascending=False)
        results['rank'] = 0
        results['prank'] = 0
        results['diff'] = 0
        results = results.reset_index()
        y_pred_sorted = sorted(results['pred'], reverse=True)
        for i, row in results.iterrows():
            true_rank = i + 1
            pred_rank = y_pred_sorted.index(row['pred']) + 1
            results.at[i, 'rank'] = true_rank
            results.at[i, 'prank'] = pred_rank
            results.at[i, 'diff'] = pred_rank - true_rank

        return results

if __name__ == '__main__':
	data = FantasyFootballData()
	x_train, x_eval, y_train, y_eval = data.get_training_data()
	x_test, y_test = data.get_test_data()
	print('Train set is {} elements.'.format(len(x_train)))
	print('Eval set is {} elements.'.format(len(x_eval)))
	print('Test set is {} elements.'.format(len(x_test)))

	print(data.get_eval_player_results())

