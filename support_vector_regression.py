import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn import svm
from fantasy_football_data import FantasyFootballData

# read training data
#df = pd.read_csv('./data/processed/training_data.csv', sep=',')
#
#columns_to_drop = ['uid', 'gid', 'week', 'player', 'name', 'position1', 'year', 'team']
#
## 2540 total rows.. 80 training = 2032, 20 testing = 508
#
## drop columns we do not need for features
#df.drop(columns=columns_to_drop, inplace=True)
#
## randomize data frame and reset the indices
#df = shuffle(df)
#df = df.reset_index()
#
## separate stats/player attributes from fantasy points
#X = [df.iloc[:, 0:24]]
#y = [df.iloc[:, 24]]
#X = np.nan_to_num(X)
#y = np.nan_to_num(y)
#X = X[0]
#y = y[0]
#
## separate into training and testing samples
#trainX = X[:2032]  # training samples
#trainY = y[:2032]  # labels for training samples
#
#testX = X[2032:]  # testing samples
#testY = y[2032:]  # labels for testing samples
data = FantasyFootballData()

# perform SVR using different kernels
for k in ['linear', 'rbf', 'poly']:
  clf = svm.SVR(kernel=k)
  clf.fit(data.x_train, data.y_train)
  confidence = clf.score(data.x_eval, data.y_eval)
  print(k, confidence)
