#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import ticker


work_dir = Path.home() / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

# df = df.sample(frac=0.2)
t0 = time()
X = df.drop('readmitted', axis=1)
y = df.loc[:, 'readmitted']

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
train_sizes = np.linspace(.1, 1.0, 5)


def learning_curves_data(estimator, X, y, cv=cv, train_sizes=train_sizes):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1,
                                                           train_sizes=train_sizes,
                                                           scoring='accuracy', shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    return train_sizes, train_scores_mean, train_scores_std, val_scores_mean, val_scores_std


model_dict = {'Logistic regression': LogisticRegression(n_jobs=-1, solver='newton-cg'),
              'Decision tree classifier': DecisionTreeClassifier(max_depth=12, random_state=42),
              'Random forest classifier': RandomForestClassifier(n_jobs=-1, random_state=42,
                                                                 max_depth=40, n_estimators=500)}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (model_name, model_instance) in zip(axes, model_dict.items()):
    means_std_list = learning_curves_data(model_instance, X, y)
    train_sizes, train_scores_means, train_scores_std, val_scores_means, val_scores_std = means_std_list
    ax.fill_between(train_sizes, train_scores_means - train_scores_std,
                    train_scores_means + train_scores_std, alpha=0.1, color='r')
    ax.fill_between(train_sizes, val_scores_means - val_scores_std,
                    val_scores_means + val_scores_std, alpha=0.1, color='b')
    ax.plot(train_sizes, train_scores_means, 'r.-', linewidth=1, label='training')
    ax.plot(train_sizes, val_scores_means, 'b.-', linewidth=1, label='validation')
    ax.legend(loc='best', fontsize=12)
    ax.set_title('Learning curve for {0:s}'.format(model_name.lower()), fontsize=14)
    ax.set_xlabel('Training examples (in units of $10^3$)', fontsize=12)
    ticks = ticker.FuncFormatter(lambda x, _: '{0:g}'.format(x*1e-3))
    ax.xaxis.set_major_formatter(ticks)
    ax.set_ylabel('Score', fontsize=12)
fig.tight_layout()
plt.savefig(work_dir / 'plots/learning_curves_plot.png', dpi=288, bbox_inches='tight')
print(f'Time elapsed: {(time() - t0):.2f} seconds')
