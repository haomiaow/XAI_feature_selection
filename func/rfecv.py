"""Recursive feature elimination for feature ranking"""

import numpy as np
from sklearn.utils.metaestimators import _safe_split
from sklearn.base import clone
from sklearn.model_selection._validation import _score
from sklearn.metrics import check_scoring

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

import time
from tqdm import tqdm

def fit_scores(estimator, X, y, ranks_desc, step_score=None, verbose = 0):
    # Initialization
    n_features = X.shape[1]
    support_ = np.ones(n_features, dtype=bool)
    scores_ = []

    while len(ranks_desc) > 0:
        # Remaining features
        features = np.arange(n_features)[support_]

        # Rank the remaining features
        estimator = clone(estimator)
        if verbose > 0:
            print("Fitting estimator with %d features." % np.sum(support_))

        estimator.fit(X[:, features], y)

        # Compute step score, insert score in the first place
        # score[i] = best i features score = first i features of ranks_desc
        scores_.insert(0, step_score(estimator, features))

        # delete the worst one
        support_[ranks_desc[-1]] = False
        ranks_desc = ranks_desc[:-1]

    return scores_


def get_n_features_to_select(estimator, X, y, ranks_desc, *,
                             cv=5, scoring=None, verbose=0, groups=None):
    if len(set(ranks_desc)) != len(ranks_desc):
        raise ValueError("Invalid Ranks")

    # Initialization
    cv = StratifiedKFold(n_splits=cv)
    scorer = check_scoring(estimator, scoring=scoring)
    n_features = X.shape[1]

    scores = []

    for train, test in cv.split(X, y, groups):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)

        scores.append(fit_scores(estimator, X_train, y_train, 
                                 ranks_desc,
            lambda estimator, features: _score(estimator, X_test[:, features], y_test, scorer)))

    scores = np.sum(scores, axis=0)
    # print(scores)
    n_features_to_select = np.argmax(scores) + 1

    return n_features_to_select


def get_n_features_to_select_grid_rf(X, y, ranks_desc, *,
                                     cv=5, scoring=None, verbose=0, groups=None):
    if len(set(ranks_desc)) != len(ranks_desc):
        raise ValueError("Invalid Ranks")

    # Initialization
    estimator = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=cv)
    scorer = check_scoring(estimator, scoring=scoring)
    n_features = X.shape[1]

    param_grid = {
        'max_depth': [3, 5, 8],
        'max_samples': 0.1*np.array(range(5, 10))
    }
    CV_rfc = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)

    scores = []
    params = []
    
    t = time.time()

    for train, test in tqdm(cv.split(X, y, groups)):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)

        CV_rfc.fit(X_train, y_train)
        param = CV_rfc.best_params_

        estimator_param = RandomForestClassifier(max_depth=param['max_depth'],
                                                 max_samples=param['max_samples'],
                                                 random_state=42)

        scores.append(fit_scores(estimator_param, X_train, y_train,
                                 ranks_desc,
            lambda estimator_param, features: _score(estimator_param, X_test[:, features], y_test, scorer)))
        params.append(param)
        
    # best_param = np.average(scores, axis=1)
    scores = np.average(scores, axis=0)
    # print(scores)
    n_features_to_select = np.argmax(scores) + 1

    #print(params)

    return n_features_to_select, params, time.time()-t