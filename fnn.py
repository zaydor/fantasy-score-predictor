"""
Authors: Isaiah Thomas and Dillon Johnson
File: fnn.py
Date: 12/18/2020
Class: CS596

Train and Test a Feed Forward Neural Network for making fantasy football predictions.
"""

from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import argparse
from fantasy_football_data import FantasyFootballData, FFEvaluation

def create_model(input_shape):
    """Set up the neural network architecture."""
    input_layer = Input(shape=input_shape)
    dense_layer_1 = Dense(4, activation='relu')(input_layer)
    dense_layer_2 = Dense(4, activation='relu')(dense_layer_1)
    dense_layer_3 = Dense(2, activation='relu')(dense_layer_2)
    output = Dense(1)(dense_layer_3)

    model = Model(inputs=input_layer, outputs=output)
    opt = SGD(learning_rate=0.001, momentum=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

    return model

def train():
    """Train a FNN model and print/plot the results."""

    data = FantasyFootballData()
    X_train, X_test, y_train, y_test = data.get_training_data(random_state=11)

    model = create_model((X_train.shape[1],))
    
    print(model.summary())
    history = model.fit(X_train, y_train, batch_size=10, epochs=200, verbose=1, validation_split=0.2)

    y_pred = model.predict(X_test)
    performances = data.get_eval_player_performances()
    performances['pred'] = y_pred

    rankings = FFEvaluation(performances).get_rankings()
    pd.set_option('display.max_rows', None)
    print(rankings)
    print('Validation score: {}'.format(r2_score(y_test, y_pred)))
    print('Validation mean squared error: {}'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
    print('Average rank error: {}'.format(np.mean(abs(rankings['diff']))))

    model.save_weights('./fnn_models/fnn')

def test(saved_model):
    """Run the specified model on the test set and print the results."""

    data = FantasyFootballData()
    X_test, y_test = data.get_test_data()
    model = create_model((X_test.shape[1],))
    model.load_weights(saved_model)
    y_pred = model.predict(X_test)

    performances = data.get_test_player_performances()
    performances['pred'] = y_pred

    rankings = FFEvaluation(performances).get_rankings()
    pd.set_option('display.max_rows', None)
    print(rankings)
    print('Average rank error: {}'.format(np.mean(abs(rankings['diff']))))
    print('R2 score: {}'.format(r2_score(y_test, y_pred)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test fnn.')
    parser.add_argument('--test', action='store', default=None)
    args = parser.parse_args()

    if args.test is not None:
        test(args.test)
    else:
        train()
