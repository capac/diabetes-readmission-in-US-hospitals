#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from time import time
import json
from sklearn.model_selection import learning_curve
from sklearn.compose import (make_column_selector,
                             make_column_transformer)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import ticker


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
    train_sizes, train_scores, val_scores = learning_curve(estimator,
                                                           X_resampled,
                                                           y_resampled,
                                                           cv=cv, n_jobs=-1,
                                                           train_sizes=tr_arr,
                                                           scoring='accuracy',
                                                           shuffle=True,
                                                           random_state=42,)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    return (train_sizes, train_scores_mean, train_scores_std,
            val_scores_mean, val_scores_std)


# Load nested parameters from JSON file
with open('params.json', 'r') as f:
    model_params = json.load(f)

model_dict = {
    'Logistic regression': LogisticRegression(
        n_jobs=-1,
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

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (model_name, model_instance) in zip(axes, model_dict.items()):
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
    ax.legend(loc='best', fontsize=12)
    ax.set_title('Learning curve for {0:s}'.format(model_name.lower()),
                 fontsize=13)
    ax.set_xlabel('Training examples (in units of $10^3$)',
                  fontsize=12)
    ticks = ticker.FuncFormatter(lambda x, _: '{0:g}'.format(x*1e-3))
    ax.xaxis.set_major_formatter(ticks)
    ax.set_ylabel('Score', fontsize=12)
fig.suptitle('Learning curves', fontsize=15)
fig.tight_layout()
plt.savefig(work_dir / 'plots/learning_curves_plot.png',
            dpi=288, bbox_inches='tight')
print(f'Time elapsed: {(time() - t0):.2f} seconds')
