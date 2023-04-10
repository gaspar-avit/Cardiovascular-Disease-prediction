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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, RocCurveDisplay
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval


### Functions ###


# --- Define dataset --- #
# # Load Dataset
dataset = pd.read_csv("./cardio_train.csv", sep=";", low_memory=False)


# # Preprocessing
# Drop 'id' column and duplicates
dataset.drop("id", axis=1, inplace=True)
dataset.drop_duplicates(inplace=True)

# # Data cleansing
# # Filter blood pressure outliers
# dataset = dataset[~((dataset["ap_hi"] > 250) | (dataset["ap_lo"] > 200)
#                     | (dataset["ap_hi"] < 0) | (dataset["ap_lo"] < 0))]

# Drop height outliers
dataset.drop(dataset[(dataset['height'] > dataset['height'].quantile(0.975)) |
                     (dataset['height'] < dataset['height'].quantile(0.025))]
             .index, inplace=True)

# Drop weight outliers
dataset.drop(dataset[(dataset['weight'] > dataset['weight'].quantile(0.975)) |
                     (dataset['weight'] < dataset['weight'].quantile(0.025))]
             .index, inplace=True)

# Fix blood pressure outliers
dataset.drop(dataset[(dataset['ap_hi'] > dataset['ap_hi'].quantile(0.975)) |
                     (dataset['ap_hi'] < dataset['ap_hi'].quantile(0.025))]
             .index, inplace=True)
dataset.drop(dataset[(dataset['ap_lo'] > dataset['ap_lo'].quantile(0.975)) |
                     (dataset['ap_lo'] < dataset['ap_lo'].quantile(0.025))]
             .index, inplace=True)


# # Data types conversion
# Convert 'age' to years
dataset["age"] = dataset["age"].map(lambda x: math.ceil(x/365))
dataset["gender"] = dataset["gender"].astype('category')
dataset["height"] = dataset["height"].astype(int)
dataset["weight"] = dataset["weight"].astype(int)
dataset["cholesterol"] = dataset["cholesterol"].astype('category')
dataset["gluc"] = dataset["gluc"].astype('category')
dataset["smoke"] = dataset["smoke"].astype(bool)
dataset["alco"] = dataset["alco"].astype(bool)
dataset["active"] = dataset["active"].astype(bool)


# # Feature Engineering
dataset["bmi"] = dataset["weight"] / (dataset["height"]/100)**2
dataset["bad_habits"] = dataset["smoke"] & dataset["alco"]


# # Prepare dataset for modelling
# Select target column
target_name = 'cardio'
data_target = dataset[target_name]
dataset = dataset.drop([target_name], axis=1)

# Get categorical features
num_cols = dataset._get_numeric_data().columns
cat_cols = list(set(dataset.columns) - set(num_cols))

# Split data into train, validation and test
X_train, X_test, Y_train, Y_test = train_test_split(dataset,
                                                    data_target,
                                                    test_size=0.1,
                                                    random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                  Y_train,
                                                  test_size=0.2,
                                                  random_state=42)


# --- Modelling --- #
# # XGBoost
# best = fmin(fn=hyperopt_xgb_score, space=space_xgb, algo=tpe.suggest, max_evals=10)
# params_xgb = space_eval(space_xgb, best)


# # CatBoost
ap_hi_col = [i[0] for i in enumerate(dataset.columns) if 'ap_hi' in i[1]][0]
ap_lo_col = [i[0] for i in enumerate(dataset.columns) if 'ap_lo' in i[1]][0]

model = cb.CatBoostClassifier(iterations=5e2, learning_rate=5e-2,
                              depth=10, use_best_model=True,
                              per_float_feature_quantization=[
                                  str(ap_hi_col)+':border_count=1024',
                                  str(ap_lo_col)+':border_count=1024'],
                              loss_function='Logloss', eval_metric='Recall',
                              early_stopping_rounds=100, verbose=200)

# Create CatBoost pools
pool_train = cb.Pool(X_train, Y_train, cat_features=cat_cols)
pool_val = cb.Pool(X_val, Y_val, cat_features=cat_cols)
pool_test = cb.Pool(X_test, cat_features=cat_cols)

# Fit model
model.fit(pool_train, eval_set=pool_val)


# --- Testing --- #
# Get predictions
predictions = model.predict(pool_test)

# Print classification report
print("\nClassification report:\n" + classification_report(Y_test, predictions))

# Plot confusion matrix
cm = confusion_matrix(Y_test, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
disp.plot()

# Plot ROC curve
RocCurveDisplay.from_predictions(Y_test, predictions)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend()
plt.show()


# Save model
model.save_model("../model.cbm")
