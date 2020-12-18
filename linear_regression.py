#
# Linear Regression Model for RB performance in fantasy football.
#

from fantasy_football_data import FantasyFootballData
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.feature_selection import RFE
import pandas as pd
import time
import matplotlib.pyplot as plt


def run_model(data, model_name, model, params):
    X_train, X_eval, y_train, y_eval = data.get_training_data(random_state=11)

    cv = GridSearchCV(estimator=model, param_grid=params, return_train_score=True)
    cv.fit(X_train, y_train)

    cv_results = pd.DataFrame(cv.cv_results_)

    print('##################################################')
    print('Results for {}'.format(model_name))
    print(cv_results)
    print('Best params: {}'.format(cv.best_params_))
    print('Test score: {}'.format(max(cv_results['mean_test_score'])))

    for param in params:
        cv_name = 'param_' + param
        plt.figure(figsize=(16,6))
        plt.plot(cv_results[cv_name], cv_results['mean_test_score'])
        plt.xlabel(param)
        plt.ylabel('score')
        plt.title('{}: {} vs Score'.format(model_name, param))
        plt.show()

    print('\n\n\n')

if __name__ == '__main__':
    data = FantasyFootballData(features=FantasyFootballData.all_features)
    #data = FantasyFootballData()

    models = {
        'Linear': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'alpha': [1, 10, 100, 150, 200, 250, 300, 350, 400, 450, 500]}),
        'Lasso': (Lasso(), {'alpha': [0.01, 0.02, 0.04, 0.06, 0.08, 0.09, 0.1, 0.11, 0.12, 0.14, 0.16]}),
        'Elastic_net': (ElasticNet(), {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]}),
    }

    for model in models:
        lm, params = models[model]
        run_model(data, model, lm, params)

    model = LinearRegression()
    model.fit(data.x_train, data.y_train)
    print('Linear regression evaluation score: {}'.format(model.score(data.x_eval, data.y_eval)))

    model = Ridge(alpha=200)
    model.fit(data.x_train, data.y_train)
    print('Ridge regression evaluation score: {}'.format(model.score(data.x_eval, data.y_eval)))

    model = Lasso(alpha=0.04)
    model.fit(data.x_train, data.y_train)
    print('Lasso regression evaluation score: {}'.format(model.score(data.x_eval, data.y_eval)))

    model = ElasticNet(alpha=0.1)
    model.fit(data.x_train, data.y_train)
    print('Elastic Net regression evaluation score: {}'.format(model.score(data.x_eval, data.y_eval)))

