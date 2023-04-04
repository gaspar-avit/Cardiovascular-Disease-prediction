# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 20:23:33 2023

@author: a780556
"""

import numpy as np
import xgboost as xgb

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval


# Parameter spacees
space_xgb = {
    'learning_rate': hp.quniform('learning_rate', 0, 0.05, 0.0001),
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    'eta': hp.quniform('eta', 0.025, 0.5, 0.005),
    'max_depth':  hp.choice('max_depth', np.arange(2, 12, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 9, 0.025),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.005),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.005),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.005),
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'tree_method': 'exact',
    'silent': 1,
    'missing': None
}

### Functions ###


def hyperopt_xgb_score(params):
    """
    Align input image text with the margins, in order to be straightened.

    :param image: Input image in the form of a PIL BW object.

    :return: Processed PIL image.
    """
    clf = xgb.XGBClassifier(**params)
    current_score = cross_val_score(clf, X_train, Y_train, cv=10).mean()
    print(current_score, params)
    return current_score


def tune_hyperparam(model, X_train, Y_train):
    """
    Align input image text with the margins, in order to be straightened.

    :param image: Input image in the form of a PIL BW object.

    :return: Processed PIL image.
    """
    if model == 'xgb':
        a
    elif model == 'lgb':
        b
    elif model == 'catboost':
        c
    else:
        ValueError('Model type not recognized. Try again with \'xgb\', \'lgb\' \
                   or \'catboost\'.')

    return best
