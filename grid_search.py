#!/usr/bin/env python

import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

home = os.environ['HOME']
work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

X = df.drop('readmitted', axis=1)
y = df.loc[:, 'readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {'max_depth': range(40, 47, 2), 'n_estimators': range(260, 321, 20)}

rf_clf = RandomForestClassifier()

rf_grid_search = GridSearchCV(rf_clf, param_grid, cv=5, return_train_score=True,
                              verbose=1, n_jobs=-1, scoring='f1')

rf_grid_search.fit(X_train, y_train)
print(f'Best parameters: {rf_grid_search.best_params_}\n')
print(f'Best estimator: {rf_grid_search.best_estimator_}\n')

cvres_df = pd.DataFrame(rf_grid_search.cv_results_)
cvres_df.sort_values(by='mean_test_score', ascending=False, inplace=True)
print(cvres_df[['mean_test_score', 'params']])
