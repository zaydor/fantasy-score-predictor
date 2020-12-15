import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# read training data
df = pd.read_csv('./data/processed/training_data.csv', sep=',')

columns_to_drop = ['uid', 'gid', 'week', 'player', 'name', 'position1', 'year', 'team']

# drop columns we do not need for features
df.drop(columns=columns_to_drop, inplace=True)

# randomize data frame and reset the indices
df = shuffle(df)
df = df.reset_index()

# separate stats/player attributes from fantasy points
X = [df.iloc[:, 0:24]]
y = [df.iloc[:, 24]]
X = np.nan_to_num(X)
y = np.nan_to_num(y)
X = X[0]
y = y[0]

# separate into training and testing samples
# 2540 total rows.. 90% to training = 2286 (200 to validate, 2086 to train), 10% to testing = 254
trainX = X[200:2286]  # training samples
trainY = y[200:2286]  # labels for training samples

validateX = X[:200]  # validation samples
validateY = y[:200]  # labels for validation samples

testX = X[2286:]  # testing samples
testY = y[2286:]  # labels for testing samples

# perform SVR using different kernels
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
prediction = []
for kernel_value in kernel_types:
  clf = svm.SVR(kernel=kernel_value)
  clf.fit(trainX, trainY)
  accuracy = clf.score(validateX, validateY)
  error = 1 - accuracy
  print(kernel_value, accuracy)
  svm_kernel_error.append(accuracy)
  prediction.append(clf.predict(testX))

# plot a prediction vs actual value graph
plt.scatter(prediction[0], testY, color='black')
x = np.linspace(-1, 30, 100)
plt.plot(x, x, color='blue')
plt.title('Fantasy Score Predictions vs Actual Score')
plt.xlabel('Prediction')
plt.ylabel('Actual Value')
plt.show()

# plot a kernel accuracy comparison graph
plt.plot(kernel_types, svm_kernel_error)
plt.title('SVR by Kernels')
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.xticks(kernel_types)
plt.show()
