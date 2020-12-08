from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from fantasy_football_data import FantasyFootballData, FFEvaluation

data = FantasyFootballData()
X_train, X_test, y_train, y_test = data.get_training_data(random_state=11)

input_layer = Input(shape=(X_train.shape[1],))
dense_layer_1 = Dense(4, activation='relu')(input_layer)
dense_layer_2 = Dense(4, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(2, activation='relu')(dense_layer_2)
output = Dense(1)(dense_layer_3)

model = Model(inputs=input_layer, outputs=output)
opt = SGD(learning_rate=0.001, momentum=0.0)
#opt = Adam(learning_rate=0.001, epsilon=1)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1, validation_split=0.2)

y_pred = model.predict(X_test)
print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))

performances = data.get_eval_player_performances()
performances['pred'] = y_pred

rankings = FFEvaluation(performances).get_rankings()
pd.set_option('display.max_rows', None)
print(rankings)
print('Average rank error: {}'.format(np.mean(abs(rankings['diff']))))

