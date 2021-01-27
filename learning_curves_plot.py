#!/usr/bin/env python

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

home = os.environ['HOME']
work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

df = df.sample(frac=0.2)
X = df.drop('readmitted', axis=1)
y = df.loc[:, 'readmitted']


def learning_curves_data(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    train_scores_means, train_scores_std = np.array([]), np.array([])
    val_scores_means, val_scores_std = np.array([]), np.array([])
    for m in range(6, len(X_train), 1000):
        model.fit(X_train.iloc[:m], y_train.iloc[:m])
        y_train_predict = model.predict(X_train.iloc[:m])
        y_val_predict = model.predict(X_val)
        train_scores = accuracy_score(y_train.iloc[:m], y_train_predict)
        train_scores_means = np.append(train_scores_means, train_scores.mean())
        train_scores_std = np.append(train_scores_std, train_scores.std())
        val_scores = accuracy_score(y_val, y_val_predict)
        val_scores_means = np.append(val_scores_means, val_scores.mean())
        val_scores_std = np.append(val_scores_std, val_scores.std())
    return train_scores_means, train_scores_std, val_scores_means, val_scores_std


model_dict = {'Logistic regression': LogisticRegression(n_jobs=-1, solver='newton-cg'),
              'Decision tree classifier': DecisionTreeClassifier(max_depth=12, random_state=42),
              'Random forest classifier': RandomForestClassifier(n_jobs=-1, random_state=42,
                                                                 max_depth=40, n_estimators=500)}

t0 = time()
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (model_name, model_instance) in zip(axes, model_dict.items()):
    means_std_list = learning_curves_data(model_instance, X, y)
    train_scores_means, train_scores_std, val_scores_means, val_scores_std = means_std_list
    print(f'train_scores_std: {train_scores_std}')
    print(f'val_scores_std: {val_scores_std}')
    train_sizes = np.arange(len(train_scores_means))
    ax.fill_between(train_sizes, train_scores_means - train_scores_std,
                    train_scores_means + train_scores_std, alpha=0.1, color='r')
    ax.fill_between(train_sizes, val_scores_means - val_scores_std,
                    val_scores_means + val_scores_std, alpha=0.1, color='b')
    ax.plot(train_sizes, train_scores_means, 'r.-', linewidth=1, label='training')
    ax.plot(train_sizes, val_scores_means, 'b.-', linewidth=1, label='validation')
    ax.legend(loc='best', fontsize=12)
    ax.set_title('Cross validation for {0:s}'.format(model_name.lower()), fontsize=14)
    ax.set_xlabel('Training set size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
fig.tight_layout()
plt.savefig(work_dir / 'plots/learning_curves_plot_with_err_bands.png', dpi=288, bbox_inches='tight')
print(f'Time elapsed: {(time() - t0):.2f} seconds')
