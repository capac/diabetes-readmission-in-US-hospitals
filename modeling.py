#!/usr/bin/env python

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from helper_funcs.helper_plots import conf_mx_heat_plot
from time import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


home = os.environ['HOME']
work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

X = df.drop('readmitted', axis=1)
y = df.loc[:, 'readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_dict = {'Logistic regression': LogisticRegression(n_jobs=-1, solver='newton-cg'),
              'Decision tree classifier': DecisionTreeClassifier(max_depth=12, random_state=42),
              'Random forest classifier': RandomForestClassifier(n_jobs=-1, random_state=42)}

t0 = time()
with open(work_dir / 'stats_output.txt', 'w') as f:
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

    # confusion matrix with plot
    print('Calculating confusion matrix values and plot...')
    for (name, model), y_pred in zip(model_dict.items(), y_pred_results):
        cm = confusion_matrix(y_test, y_pred)
        conf_mx_heat_plot(cm, name, work_dir / 'plots')
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
        axes.plot(fpr, tpr, marker='.', ms=8,
                  label='Model: {0:s}, Regression (area = {1:.4f})'.format(name.lower(), model_roc_auc))
        axes.plot([0, 1], [0, 1], 'r--')
        axes.set_xlim([-0.02, 1.0])
        axes.set_ylim([0.0, 1.02])
        axes.set_xlabel('False Positive Rate', fontsize=14)
        axes.set_ylabel('True Positive Rate', fontsize=14)
        axes.set_title('Receiver operating characteristic for {0:s} model'.format(name.lower()), fontsize=16)
        axes.legend(loc="lower right")
        # plt.grid(True, linestyle='--')
        plot_file = '_'.join(name.split(' ')).lower()+'_auc.png'
        plt.savefig(work_dir / 'plots' / plot_file, dpi=288, bbox_inches='tight')

    # cross validation Brier score
    def display_scores(model, scores):
        f.writelines(f'Cross-validation Brier score for the {model.lower()} model:\n')
        # f.writelines(f'Scores: {scores}')
        f.writelines(f'Average Brier score: {scores.mean():.4f}\n')
        f.writelines(f'Standard devation: {scores.std():.4f}\n')
        f.writelines('\n')

    print('Calculating average Brier score ...')
    for (name, model), y_pred in zip(model_dict.items(), y_pred_results):
        scores = cross_val_score(model, y_pred.reshape(-1, 1), y_test,
                                 scoring='neg_brier_score', cv=6, n_jobs=-1)
        display_scores(name, -scores)

print(f'Time elapsed: {(time() - t0):.2f} seconds')
