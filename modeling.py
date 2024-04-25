#!/usr/bin/env python

from time import time
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.compose import (make_column_selector,
                             make_column_transformer)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (train_test_split,
                                     cross_validate,
                                     cross_val_score)
from imblearn.under_sampling import RandomUnderSampler
from helper_funcs.helper_plots import (conf_mx_heat_plot,
                                       roc_curve_plot_with_auc)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
)

work_dir = (
    Path.home() / "Programming/Python/machine-learning-exercises/"
    "uci-ml-repository/diabetes-in-130-US-hospitals"
)
df = pd.read_csv(work_dir / "data/df_encoded.csv", low_memory=False)
num_list = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses", "service_use",
    "readmitted",
]
cat_list = list(set(df.columns) - set(num_list))

df[cat_list] = df[cat_list].astype("object")

# selecting input and label features
X = df.drop("readmitted", axis=1)
y = df["readmitted"].copy()
# generating training and testing data (30% of total data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0,
                                                    stratify=y,)


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

# load nested parameters from JSON file
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
            random_state=model_params['params_svc']['random_state'],
            ),
        'AdaBoost classifier': AdaBoostClassifier(
            algorithm='SAMME',
            n_estimators=model_params['params_ad']['n_estimators'],
            learning_rate=model_params['params_ad']['learning_rate'],
            random_state=model_params['params_ad']['random_state'],
            ),
        'Gradient boosting classifier': GradientBoostingClassifier(
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

# calculating balanced accuracy, confusion matrix, classification report
# roc curve and auc values, and average Brier score
t0 = time()
with open(work_dir / "stats_output.txt", "w") as f:
    rus = RandomUnderSampler(sampling_strategy='majority',
                             random_state=0)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_pp,
                                                            y_train)
    cm_dict = {}
    print("Calculating balanced accuracy...")
    for name, model in model_dict.items():
        cv_results = cross_validate(model, X_train_resampled,
                                    y_train_resampled,
                                    scoring="balanced_accuracy",
                                    return_train_score=True,
                                    return_estimator=True, n_jobs=-1,
                                    error_score="raise",)
        f.writelines(
            f"Training accuracy mean ± std. dev. for {name.lower()}: "
            f"{np.round(cv_results['test_score'].mean(), 4)} ± "
            f"{np.round(cv_results['test_score'].std(), 4)}"
            f"\n"
        )
        scores = []
        for cv_model in cv_results["estimator"]:
            y_test_pp = cv_model.predict(X_test_pp)
            scores.append(balanced_accuracy_score(y_test, y_test_pp))
        f.writelines(
            f"Testing accuracy mean ± std. dev. for {name.lower()}: "
            f"{np.round(np.mean(scores), 4)} ± "
            f"{np.round(np.std(scores), 4)}"
            f"\n\n"
        )
    f.writelines("\n")
    print("Calculating confusion matrix...")
    for name, model in model_dict.items():
        # confusion matrix with plot
        clf = model.fit(X_train_resampled, y_train_resampled)
        y_test_pp = clf.predict(X_test_pp)
        cm = confusion_matrix(y_test, y_test_pp,)
        f.writelines(f"Confusion matrix on {name.lower()} model: \n{cm}\n")
        cm_dict[name] = cm
        f.writelines("\n")
        del clf
    conf_mx_heat_plot(cm_dict, work_dir)
    f.writelines("\n")

    # classification report
    print("Calculating precision, recall, F-measure and support...")
    for name, model in model_dict.items():
        clf = model.fit(X_train_resampled, y_train_resampled)
        y_test_pp = clf.predict(X_test_pp)
        class_report = classification_report(y_test, y_test_pp,
                                             digits=4)
        f.writelines(
            f"Precision, recall, F-measure and support on "
            f"the {name.lower()} model: \n{class_report}\n"
        )
        del clf
    f.writelines("\n")

    # roc curve
    print("Calculating ROC plot...")
    rates_dict = {}
    for name, model in model_dict.items():
        clf = model.fit(X_train_resampled, y_train_resampled)
        y_test_pp = clf.predict_proba(X_test_pp)[:, 1]
        model_roc_auc = roc_auc_score(y_test, y_test_pp)
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pp)
        f.writelines(f"Model: {name.title()}\n"
                     f"FPR: {len(fpr)}\nTPR: {len(tpr)}\n")
        f.writelines(f"Number of thresholds: {len(thresholds)}\n")
        rates_dict[name] = [fpr, tpr, model_roc_auc]
        f.writelines("\n")
        del clf
    roc_curve_plot_with_auc(rates_dict, work_dir)

    # cross validation average Brier score
    def display_scores(model, scores):
        f.writelines(
            f"Cross-validated average Brier score for {model.lower()}: "
            f"{np.round(scores.mean(), 4)} ± "
            f"{np.round(scores.std(), 4)}"
            f"\n\n"
        )

    print("Calculating cross-validated average Brier score...")
    for name, model in model_dict.items():
        scores = cross_val_score(model, X_train_resampled,
                                 y_train_resampled,
                                 scoring="neg_brier_score",
                                 cv=6, n_jobs=-1,)
        display_scores(name, -scores)

print(f"Time elapsed: {(time() - t0):.2f} seconds")
