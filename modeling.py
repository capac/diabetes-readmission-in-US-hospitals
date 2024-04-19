#!/usr/bin/env python

from time import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import (make_column_selector,
                             make_column_transformer)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import (train_test_split,
                                     cross_validate,
                                     cross_val_score)
from imblearn.under_sampling import RandomUnderSampler
from helper_funcs.helper_plots import (conf_mx_heat_plot,
                                       roc_curve_plot_with_auc)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
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
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "service_use",
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
    num_pipeline = make_pipeline(MinMaxScaler())
    cat_pipeline = make_pipeline(OneHotEncoder(
        handle_unknown="infrequent_if_exist"),)

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

# testing several data science algorithms
model_dict = {
    "Logistic regression": LogisticRegression(n_jobs=-1, C=1e2,
                                              solver="newton-cholesky",),
    "Decision tree classifier": DecisionTreeClassifier(max_depth=16,
                                                       random_state=0,),
    "Random forest classifier": RandomForestClassifier(n_jobs=-1,
                                                       random_state=0,
                                                       max_depth=16,
                                                       n_estimators=160,),
    "GBClassifier": GradientBoostingClassifier(random_state=0),
    "HistogramGBClassifier": HistGradientBoostingClassifier(random_state=0),
    "XGBClassifier:": XGBClassifier(random_state=0),
}

# SMOTE: Synthetic Minority Over-sampling Technique
# When dealing with mixed data type such as continuous and categorical
# features, none of the presented methods (apart of the class
# RandomOverSampler) can deal with the categorical features.
# https://imbalanced-learn.org/stable/over_sampling.html#smote-variants

t0 = time()
with open(work_dir / "stats_output.txt", "w") as f:
    rus = RandomUnderSampler(sampling_strategy="majority",
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
            f"Training accuracy mean +/- std. dev. for {name.lower()}: "
            f"{cv_results['test_score'].mean():.3f} +/- "
            f"{cv_results['test_score'].std():.3f}"
            f"\n"
        )
        scores = []
        for cv_model in cv_results["estimator"]:
            scores.append(balanced_accuracy_score(y_test,
                                                  cv_model.predict(X_test_pp)))
        f.writelines(
            f"Testing accuracy mean +/- std. dev. for {name.lower()}: "
            f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
            f"\n\n"
        )
    f.writelines("\n")
    print("Calculating confusion matrix...")
    for name, model in model_dict.items():
        # confusion matrix with plot
        clf = model.fit(X_train_resampled, y_train_resampled)
        cm = confusion_matrix(y_test, clf.predict(X_test_pp),)
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
        class_report = classification_report(y_test,
                                             clf.predict(X_test_pp),
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
        model_roc_auc = roc_auc_score(y_test, clf.predict(X_test_pp))
        fpr, tpr, thresholds = roc_curve(y_test,
                                         clf.predict_proba(X_test_pp)[:, 1])
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
            f"Cross-validation Brier score for "
            f"the {model.lower()} model:\n"
        )
        # f.writelines(f'Scores: {scores}')
        f.writelines(f"Average Brier score: {scores.mean():.4f}\n")
        f.writelines(f"Standard devation: {scores.std():.4f}\n")
        f.writelines("\n")

    print("Calculating average Brier score...")
    for name, model in model_dict.items():
        clf = model.fit(X_train_resampled, y_train_resampled)
        scores = cross_val_score(
            model, clf.predict(X_test_pp).reshape(-1, 1),
            y_test, scoring="neg_brier_score",
            cv=6, n_jobs=-1,
        )
        del clf
        display_scores(name, -scores)

print(f"Time elapsed: {(time() - t0):.2f} seconds")
