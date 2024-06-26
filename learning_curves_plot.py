#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import json
from time import time
from datetime import timedelta
from sklearn.model_selection import learning_curve
from sklearn.compose import (make_column_selector,
                             make_column_transformer)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import ticker

plt.style.use("style_files/lineplot-style.mplstyle")

work_dir = Path.home() / 'Programming/Python/machine-learning-exercises/'\
                         'uci-ml-repository/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

# df = df.sample(frac=0.2)
num_list = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses", "service_use",
    "readmitted",
]
cat_list = list(set(df.columns) - set(num_list))
df[cat_list] = df[cat_list].astype("object")

X = df.drop('readmitted', axis=1)
y = df['readmitted'].copy()
tr_arr = np.linspace(0.1, 1.0, 5)
t0 = time()


def preprocessing_data(X):
    # standardize numeric data and generate one-hot encoded data features
    num_pipeline = make_pipeline(StandardScaler())
    cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"),)

    # preprocessing pipeline
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include="object"),),
        sparse_threshold=0,)

    # output dataframe from preprocessing pipeline
    X_pp = preprocessing.fit_transform(X)
    columns = preprocessing.get_feature_names_out()
    return pd.DataFrame(X_pp, columns=columns,
                        index=X.index,)


X_pp = preprocessing_data(X)
rus = RandomUnderSampler(sampling_strategy='majority', random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_pp, y)


def learning_curves_data(estimator, X, y, cv=5, train_sizes=tr_arr):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_resampled, y_resampled,
        cv=cv, n_jobs=-1, train_sizes=tr_arr,
        scoring='accuracy', shuffle=True, random_state=42,)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    return (train_sizes, train_scores_mean, train_scores_std,
            val_scores_mean, val_scores_std)


# Load nested parameters from JSON file
with open('params.json', 'r') as f:
    model_params = json.load(f)

########################################
use_models_that_prioritize_recall = True
########################################
if use_models_that_prioritize_recall:
    model_dict = {
        'SVC': SVC(
            probability=True,
            C=model_params['params_svc']['C'],
            gamma=model_params['params_svc']['gamma'],
            random_state=model_params['params_svc']['random_state'],
            ),
        'AdaBoost Classifier': AdaBoostClassifier(
            algorithm='SAMME',
            n_estimators=model_params['params_ad']['n_estimators'],
            learning_rate=model_params['params_ad']['learning_rate'],
            random_state=model_params['params_ad']['random_state'],
            ),
        'Gradient Boosting Classifier': GradientBoostingClassifier(
            n_estimators=model_params['params_gb']['n_estimators'],
            learning_rate=model_params['params_gb']['learning_rate'],
            random_state=model_params['params_gb']['random_state'],
        ),
    }
else:
    model_dict = {
        'Logistic regression': LogisticRegression(
            n_jobs=-1, max_iter=4000,
            C=model_params['params_lr']['C'],
            solver='newton-cholesky'
            ),
        'Decision tree classifier': DecisionTreeClassifier(
            max_depth=model_params['params_dt']['max_depth'],
            min_samples_split=model_params['params_dt']['min_samples_split'],
            random_state=model_params['params_dt']['random_state'],
            ),
        'Random forest classifier': RandomForestClassifier(
            n_jobs=-1,
            n_estimators=model_params['params_rf']['n_estimators'],
            max_depth=model_params['params_rf']['max_depth'],
            random_state=model_params['params_rf']['random_state'],
            )
        }

# plot settings
row_length_in_px = 12
column_length_in_px = 4
nrows = 1
ncols = 3

fig, axes = plt.subplots(nrows, ncols, figsize=(row_length_in_px,
                                                column_length_in_px))
for ax, (model_name, model_instance) in zip(axes.flatten(),
                                            model_dict.items()):
    means_std_list = learning_curves_data(model_instance, X, y)
    (train_sizes, train_scores_means, train_scores_std,
     val_scores_means, val_scores_std) = means_std_list
    ax.fill_between(train_sizes, train_scores_means - train_scores_std,
                    train_scores_means + train_scores_std,
                    alpha=0.1, color='r')
    ax.fill_between(train_sizes, val_scores_means - val_scores_std,
                    val_scores_means + val_scores_std, alpha=0.1, color='b')
    ax.plot(train_sizes, train_scores_means, 'r.-',
            linewidth=1, label='Training')
    ax.plot(train_sizes, val_scores_means, 'b.-',
            linewidth=1, label='Validation')
    ax.legend(loc='best', fontsize=9)
    if len(model_name) > 3:
        ax.set_title(f'ROC curve for {model_name.lower()}', fontsize=10)
    else:
        ax.set_title(f'ROC curve for {model_name.upper()}', fontsize=10)
    ticks = ticker.FuncFormatter(lambda x, _: '{0:g}'.format(x*1e-3))
    ax.xaxis.set_major_formatter(ticks)
    ax.set_xlabel('Training examples (in units of $10^3$)', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_ylim([0.532, 0.648])
    plt.setp(ax.get_xticklabels(), fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
fig.suptitle('Learning curves', fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig(work_dir / 'plots/learning_curves_plot.png',
            bbox_inches='tight')

seconds_rounded = round(time() - t0)
time_elapsed_in_secs = str(timedelta(seconds=seconds_rounded)).split(':')
formatted_time = '{}h {}m {}s'.format(*time_elapsed_in_secs)
print(f'Time elapsed: {formatted_time}.')
