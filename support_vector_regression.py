import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from fantasy_football_data import FantasyFootballData

data = FantasyFootballData()

# perform SVR using different kernels
for k in ['linear', 'rbf', 'poly']:
  clf = svm.SVR(kernel=k)
  clf.fit(data.x_train, data.y_train)
  confidence = clf.score(data.x_eval, data.y_eval)
  print(k, confidence)

# perform SVR using different kernels
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
prediction = []
for kernel_value in kernel_types:
  clf = svm.SVR(kernel=kernel_value)
  clf.fit(data.x_train, data.y_train)
  accuracy = clf.score(data.x_eval, data.y_eval)
  error = 1 - accuracy
  print(kernel_value, accuracy)
  svm_kernel_error.append(accuracy)
  prediction.append(clf.predict(data.x_eval))

# plot a prediction vs actual value graph
plt.scatter(prediction[0], data.y_eval, color='black')
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
