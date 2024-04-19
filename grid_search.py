#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import (make_column_selector,
                             make_column_transformer)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)


def preprocessing_data(X_train, X_test):
    # standardize numeric data and generate one-hot encoded data features
    num_pipeline = make_pipeline(StandardScaler())
    cat_pipeline = make_pipeline(OneHotEncoder(
        handle_unknown="ignore"),)

    # preprocessing pipeline
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include="object"),),
        sparse_threshold=0,)

    # output dataframe from preprocessing pipeline
    X_train_pp = preprocessing.fit_transform(X_train)
    pp_columns = preprocessing.get_feature_names_out()
    X_test_pp = preprocessing.transform(X_test)
    X_train_pp = pd.DataFrame(X_train_pp, columns=pp_columns,
                              index=X_train.index,)
    X_test_pp = pd.DataFrame(X_test_pp, columns=pp_columns,
                             index=X_test.index,)
    return X_train_pp, X_test_pp


X_train_pp, X_test_pp = preprocessing_data(X_train, X_test)


rus = RandomUnderSampler(sampling_strategy="majority", random_state=0)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_pp, y_train)


param_grid = {'max_depth': np.arange(4, 20, 4),
              'n_estimators': np.arange(10, 200, 50),
              'random_state': [42]}

rf_clf = RandomForestClassifier()

rf_grid_search = GridSearchCV(rf_clf, param_grid, cv=5,
                              return_train_score=True,
                              verbose=1, n_jobs=-1,
                              scoring='f1')

rf_grid_search.fit(X_train_resampled, y_train_resampled)
print(f'Best parameters: {rf_grid_search.best_params_}\n')
print(f'Best estimator: {rf_grid_search.best_estimator_}\n')

cvres_df = pd.DataFrame(rf_grid_search.cv_results_)
cvres_df.sort_values(by='mean_test_score', ascending=False, inplace=True)
print(cvres_df[['mean_test_score', 'params']])
