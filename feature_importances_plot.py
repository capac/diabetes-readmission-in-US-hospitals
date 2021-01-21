#!/usr/bin/env python

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

home = os.environ['HOME']
work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

X = df.drop('readmitted', axis=1)
y = df.loc[:, 'readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_clf = RandomForestClassifier(n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)
std_err = np.std([tree.feature_importances_ for tree in rf_clf.estimators_], axis=0)

sorted_features_df = pd.DataFrame({'col_name': rf_clf.feature_importances_, 'std_err': std_err},
                                  index=X.columns).sort_values(by='col_name', ascending=False)

fig, axes = plt.subplots(figsize=(9, 5))
axes.bar(sorted_features_df.index, sorted_features_df['col_name'],
         color=plt.cm.Paired.colors, edgecolor='k', yerr=sorted_features_df['std_err'])
plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=10)
plt.setp(axes.get_yticklabels(), fontsize=10)
axes.set_title('Feature importances', fontsize=14)
plt.savefig('plots/feature_importances.png', dpi=288, bbox_inches='tight')
