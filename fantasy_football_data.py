#
# Helper methods to get consistent training and test data.
#

import pandas as pd
from sklearn.preprocessing import StandardScaler
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
        self.df = pd.read_csv(csv)

        # Just get rid of any rows with blank cells.
        self.df.dropna(inplace=True)

        # Split up and preprocess the data.
        y = self.df['fp']
        x = self.df[features]
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    def get_training_data(self, random_state=None):
        return train_test_split(self.x_train, self.y_train, random_state=random_state)

    def get_test_data(self):
        return self.x_test, self.y_test

