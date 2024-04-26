#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import json
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import SVC

work_dir = Path.home() / 'Programming/Python/machine-learning-exercises/'\
                         'uci-ml-repository/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

num_list = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses", "service_use",
    "readmitted",
]
cat_list = list(set(df.columns) - set(num_list))

df[cat_list] = df[cat_list].astype("object")

X = df.drop('readmitted', axis=1)
y = df['readmitted'].copy()

rus = RandomUnderSampler(sampling_strategy="majority", random_state=0)
X_train_resampled, y_train_resampled = rus.fit_resample(X, y)

# load nested parameters from JSON file
try:
    with open('params.json', 'r') as f:
        model_params = json.load(f)
except FileNotFoundError:
    model_params = {}

########################################
use_models_that_prioritize_recall = True
########################################
if use_models_that_prioritize_recall:
    model_dict = {
        'SVC': SVC(probability=True),
        'AdaBoost Classifier': AdaBoostClassifier(algorithm='SAMME'),
        'Gradient Boosting Classifier': GradientBoostingClassifier(),
    }
    # support vector classifier parameters
    svc_param_grid = {'C': np.logspace(-1, 2, 4),
                      'gamma': [0.1, 0.2, 0.5, 1.0],
                      'random_state': [42]}
    svc_clf = SVC(probability=True)

    # adaboost classifier parameters
    ad_param_grid = {'n_estimators': [50, 100, 200],
                     'learning_rate': [0.1, 0.5, 1.0],
                     'random_state': [42]}
    ad_clf = AdaBoostClassifier(algorithm='SAMME')

    # gradient boosting classifier parameters
    gb_param_grid = {'n_estimators': [5, 10, 20],
                     'learning_rate': [0.05, 0.1, 0.5],
                     'random_state': [42]}
    gb_clf = GradientBoostingClassifier()

    param_grid_dict = {'svc': svc_param_grid,
                       'ad': ad_param_grid,
                       'gb': gb_param_grid}
    clf_list = [svc_clf, ad_clf, gb_clf]
else:
    # logistic regression parameters
    lr_param_grid = {'C': np.logspace(-1, 1, 3),
                     'random_state': [42]}
    lr_clf = LogisticRegression(max_iter=4000)

    # decision tree parameters
    dt_param_grid = {'max_depth': np.arange(4, 25, 4),
                     'min_samples_split': [5, 10, 20, 50,],
                     'random_state': [42]}
    dt_clf = DecisionTreeClassifier()

    # random forest parameters
    rf_param_grid = {'n_estimators': [50, 100, 200],
                     'max_depth': np.arange(4, 25, 4),
                     'random_state': [42]}
    rf_clf = RandomForestClassifier()

    param_grid_dict = {'lr': lr_param_grid,
                       'dt': dt_param_grid,
                       'rf': rf_param_grid}
    clf_list = [lr_clf, dt_clf, rf_clf]


def calculate_best_parameters(clf, param_grid):
    grid_search = GridSearchCV(clf, param_grid, cv=5,
                               return_train_score=True,
                               verbose=1, n_jobs=-1,
                               scoring='balanced_accuracy')
    # using entire dataset for grid search
    grid_search.fit(X_train_resampled, y_train_resampled)
    print(f'Best parameters for {clf.__class__.__name__}: '
          f'{grid_search.best_params_}')
    print(f'Best estimator: {grid_search.best_estimator_}\n')

    return grid_search.best_params_


with open('params.json', 'w') as f:
    for clf, (name, param) in zip(clf_list, param_grid_dict.items()):
        best_params = calculate_best_parameters(clf, param)
        for key, val in best_params.items():
            if isinstance(val, np.int64):
                best_params[key] = int(val)
        model_params.setdefault(f'params_{name}', {}).update(best_params)
    json.dump(model_params, f, indent=4,)
