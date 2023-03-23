# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:08:26 2023

@author: a780556
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval


### Functions ###


## Load Dataset
dataset = pd.read_csv("./cardio_train.csv", sep=";", low_memory=False)


## Preprocessing
# Drop 'id' column and duplicates
dataset.drop("id", axis=1, inplace=True)
dataset.drop_duplicates(inplace=True)

# Filter blood pressure outliers
dataset = dataset[~((dataset["ap_hi"] > 250) | (dataset["ap_lo"] > 200)
                    | (dataset["ap_hi"] < 0) | (dataset["ap_lo"] < 0))]

# Convert 'age' to years
dataset["age"] = dataset["age"].map(lambda x: math.ceil(x/365))

# Feature Engineering
dataset["bmi"] = dataset["weight"] / (dataset["height"]/100)**2


## Prepare dataset for modelling
# Select target column
target_name = 'cardio'
data_target = dataset[target_name]
dataset = dataset.drop([target_name], axis=1)

# Split data into train and validation
X_train, X_val, Y_train, Y_test = train_test_split(dataset,
                                                   data_target,
                                                   test_size=0.2,
                                                   random_state=42)


## Modelling
# XGBoost
best = fmin(fn=hyperopt_xgb_score, space=space_xgb, algo=tpe.suggest, max_evals=10)
params_xgb = space_eval(space_xgb, best)
