# Evaluation of Machine Learning Regression Models to Predict Fantasy Football Performance
Authors: Isaiah Dorado and Dillon Johnson
Class: CS596 Machine Learning, Fall 2020
Date: 12/18/2020


# About
Fantasy Football assigns points to real life football
players based on how they perform in game. Fantasy managers
form a virtual team and compete head to head to see whose team
can earn the most points each week. Machine Learning could be
used to predict player performance, giving an edge to fantasy
managers when selecting players. In this project, we develop and
evaluate multiple machine learning regression models to see
which one delivers the best predictions.

This repo includes 4 models:
1) Linear Regression (including ridge, elastic-net, and lasso)
2) Polynomial Regression
3) Support Vector Regression
4) Feed-Forward Neural Network

Offensive data from: Armchair Analysis https://www.armchairanalysis.com

Defensive data from: Stathead https://stathead.com/football/

# Quick Start
## Requirements
- Python 3.8.0
- All packages in requirements.txt

## Environment
We used Ubuntu 18.04 and recommend similar. We expect most Linux distributions should "just work."
This project should run on Windows since it is python based, but we did not test this and your
mileage may vary.

## Recommendations
We recommend creating a virtual environment with virtualenv to install the required packages.
```
virtualenv -p python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Files
- Data
  - data/raw contains the unprocessed csv's.
  - data/processed contains the merged and processed data used for training.
  - fantasy_football_data.py is a wrapper around our dataset.
- Models
  - fnn.py
  - linear_regression.py
  - polynomial_regression.py
  - support_vector_regression.py
- Visualizations
  - visualize_data.py

## Data
The raw data can be processed with the following command:
```
python process_raw_data.py
```

To visualize the data:
```
python visualize_data.py
```

## Training the models
To run the different regression models on our processed running back data,
you have to run each machine learning model file individually.

Linear, Ridge, Lasso, and Elastic Net:
```
python linear_regression.py
```

Ridge Regression with polynomial features:
```
python polynomial_regression.py
```

Support Vector Regression:
```
python support_vector_regression.py
```

Feed Forward Neural Network:
```
python fnn.py
```

## Testing
Our evaluation showed that the FNN performed best. Thus, the FNN model is the only one we spent time supporting running the tests on. Use the following command to load the final FNN model and run it against the test set.
```
python fnn.py --test fnn_models/fnn
```

