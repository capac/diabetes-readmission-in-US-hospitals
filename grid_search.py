#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import json
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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

# logistic regression parameters
lr_param_grid = {'C': np.logspace(-2, 2, 5),
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
                               scoring='f1')
    # using entire dataset for grid search
    grid_search.fit(X_train_resampled, y_train_resampled)
    print(f'Best parameters for {clf.__class__.__name__}: '
          f'{grid_search.best_params_}')
    print(f'Best estimator: {grid_search.best_estimator_}\n')

    return grid_search.best_params_


with open('params.json', 'w') as f:
    parameters_dict = {}
    for clf, (name, param) in zip(clf_list, param_grid_dict.items()):
        best_params = calculate_best_parameters(clf, param)
        for key, val in best_params.items():
            if isinstance(val, np.int64):
                best_params[key] = int(val)
        parameters_dict[f'params_{name}'] = best_params
    json.dump(parameters_dict, f, indent=4,)
