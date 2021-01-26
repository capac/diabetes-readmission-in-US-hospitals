#!/usr/bin/env python

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

home = os.environ['HOME']
work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

df = df.sample(frac=0.4)
X = df.drop('readmitted', axis=1)
y = df.loc[:, 'readmitted']


def learning_curves_data(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    train_scores_means, train_scores_std = [], []
    val_scores_means, val_scores_std = [], []
    for m in range(6, len(X_train), 500):
        model.fit(X_train.iloc[:m], y_train.iloc[:m])
        y_train_predict = model.predict(X_train.iloc[:m])
        y_val_predict = model.predict(X_val)
        train_scores = cross_val_score(model, X_train.iloc[:m], y_train_predict, cv=3, scoring='accuracy')
        train_scores_means.append(train_scores.mean())
        train_scores_std.append(train_scores.std())
        val_scores = cross_val_score(model, X_val, y_val_predict, cv=3, scoring='accuracy')
        val_scores_means.append(val_scores.mean())
        val_scores_std.append(val_scores.std())
    return train_scores_means, train_scores_std, val_scores_means, val_scores_std


model_dict = {'Logistic regression': LogisticRegression(n_jobs=-1, solver='newton-cg'),
              'Decision tree classifier': DecisionTreeClassifier(max_depth=12, random_state=42),
              'Random forest classifier': RandomForestClassifier(n_jobs=-1, random_state=42,
                                                                 max_depth=40, n_estimators=500)}


fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (model_name, model_instance) in zip(axes, model_dict.items()):
    means_std_list = learning_curves_data(model_instance, X, y)
    train_scores_means, train_scores_std, val_scores_means, val_scores_std = means_std_list
    x_arr = np.arange(len(train_scores_means))
    ax.errorbar(x_arr, train_scores_means, yerr=train_scores_std, linewidth=2, label='training')
    ax.errorbar(x_arr, val_scores_means, yerr=val_scores_std, linewidth=2, label='validation')
    ax.legend(loc='best', fontsize=12)
    ax.set_title('Cross validation for {0:s}'.format(model_name.lower()), fontsize=14)
    ax.set_xlabel('Training set size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
fig.tight_layout()
plt.savefig(work_dir / 'plots/learning_curves_err_plot.png', dpi=288, bbox_inches='tight')
