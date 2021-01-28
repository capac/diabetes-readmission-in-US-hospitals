#!/usr/bin/env python

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    train_errors, val_errors = [], []
    for m in range(5, len(X_train), 500):
        model.fit(X_train.iloc[:m], y_train.iloc[:m])
        y_train_predict = model.predict(X_train.iloc[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(accuracy_score(y_train.iloc[:m], y_train_predict))
        val_errors.append(accuracy_score(y_val, y_val_predict))
    return train_errors, val_errors


model_dict = {'Logistic regression': LogisticRegression(n_jobs=-1, solver='newton-cg'),
              'Decision tree classifier': DecisionTreeClassifier(max_depth=12, random_state=42),
              'Random forest classifier': RandomForestClassifier(n_jobs=-1, random_state=42,
                                                                 max_depth=40, n_estimators=500)}


fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (model_name, model_instance) in zip(axes, model_dict.items()):
    train_errors, val_errors = learning_curves_data(model_instance, X, y)
    ax.plot(train_errors, 'r.-', linewidth=2, label='training')
    ax.plot(val_errors, 'b.-', linewidth=2, label='validation')
    ax.legend(loc='best', fontsize=12)
    ax.set_title('Accuracy score for {0:s}'.format(model_name.lower()), fontsize=14)
    ax.set_xlabel('Training set size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
fig.tight_layout()
plt.savefig(work_dir / 'plots/learning_curves_plot.png', dpi=288, bbox_inches='tight')
