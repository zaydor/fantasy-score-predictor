#
# Linear Regression Model for RB performance in fantasy football.
#

from fantasy_football_data import PolyFantasyFootballData
from sklearn.linear_model import Ridge
from sklearn.model_selection import  learning_curve
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    degrees = [1,2,3]

    scores = []
    for degree in degrees:
        data = PolyFantasyFootballData(degree=degree)
        model = Ridge(alpha=200)
        train_sizes, train_scores, test_scores = \
            learning_curve(model, data.x_train, data.y_train, random_state=11)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, test_scores_mean)
        model.fit(data.x_train, data.y_train)
        print('Evaluation score: {}'.format(model.score(data.x_eval, data.y_eval)))

    plt.title('Ridge Regression with Polynomial Features')
    plt.legend(['1st degree', '2nd degree', '3rd degree'])
    plt.xlabel('Training samples')
    plt.ylabel('Score')
    plt.show()


