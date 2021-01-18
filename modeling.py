#!/usr/bin/env python

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_auc_score, roc_curve


home = os.environ['HOME']
work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

X = df.drop('readmitted', axis=1)
y = df.loc[:, 'readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_dict = {'Logistic regression': LogisticRegression(n_jobs=-1, solver='newton-cg'),
              'Decision tree classifier': DecisionTreeClassifier(max_depth=16),
              'Random forest classifier': RandomForestClassifier(n_jobs=-1, n_estimators=200)}


t0 = time()
with open('stats_output.txt', 'w') as f:
    # model accuracy
    y_pred_results = []
    y_pred_proba_results = []
    print('Calculating accuracy...')
    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_results.append(y_pred)
        y_pred_proba = model.predict_proba(X_test)
        y_pred_proba_results.append(y_pred_proba)
        f.writelines(f'Accuracy of the {name.lower()} on test set: {model.score(X_test, y_test):.4f}\n')
    f.writelines('\n')

    # mean squared error on test set
    print('Calculating RMSE...')
    for (name, model), y_pred in zip(model_dict.items(), y_pred_results):
        mse = mean_squared_error(y_test, y_pred)
        f.writelines(f'RMSE on test set with {name.lower()} model: {np.sqrt(mse):.4f}\n')
    f.writelines('\n')

    # confusion matrix
    print('Calculating confusion matrix values...')
    for (name, model), y_pred in zip(model_dict.items(), y_pred_results):
        cm = confusion_matrix(y_test, y_pred)
        f.writelines(f'Confusion matrix on {name.lower()} model: \n{cm}\n')
    f.writelines('\n')

    # classification report
    print('Calculating precision, recall, F-measure and support...')
    for (name, model), y_pred in zip(model_dict.items(), y_pred_results):
        class_report = classification_report(y_test, y_pred, digits=4)
        f.writelines(f'Precision, recall, F-measure and support on the {name.lower()} model: \n{class_report}\n')
    f.writelines('\n')

    # roc curve
    print('Calculating ROC plot...')
    for (name, model), y_pred, y_pred_proba in zip(model_dict.items(), y_pred_results, y_pred_proba_results):
        model_roc_auc = roc_auc_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
        f.writelines(f'Model: {name.title()}\nFPR: {len(fpr)}\nTPR: {len(tpr)}\n')
        f.writelines(f'Number of thresholds: {len(thresholds)}\n')
        f.writelines('\n')
        fig, axes = plt.subplots(figsize=(10, 8))
        axes.plot(fpr, tpr, marker='.', ms=6,
                  label='Model: {0:s}, Regression (area = {1:.4f})'.format(name.lower(), model_roc_auc))
        axes.plot([0, 1], [0, 1], 'r--')
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])
        axes.set_xlabel('False Positive Rate')
        axes.set_ylabel('True Positive Rate')
        axes.set_title('Receiver operating characteristic for {0:s} model'.format(name.lower()))
        axes.legend(loc="lower right")
        # plt.grid(True, linestyle='--')
        name = '_'.join(name.split(' ')).lower()
        plt.savefig(name+'_auc.png', dpi=288, bbox_inches='tight')

    # cross validation score
    def display_scores(model, scores):
        f.writelines(f'Cross validation for the {model.lower()} model:\n')
        # f.writelines(f'Scores: {scores}')
        f.writelines(f'Mean: {scores.mean():.4f}\n')
        f.writelines(f'Standard devation: {scores.std():.4f}\n')
        f.writelines('\n')

    print('Calculating cross validation score...')
    for (name, model), y_pred in zip(model_dict.items(), y_pred_results):
        scores = cross_val_score(model, y_pred.reshape(-1, 1), y_test,
                                 scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
        display_scores(name, -scores)

print(f'Time elapsed: {(time() - t0):.2f} seconds')
