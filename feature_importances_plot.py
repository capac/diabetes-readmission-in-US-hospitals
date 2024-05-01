#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

plt.style.use("style_files/barplot-style.mplstyle")

work_dir = (
    Path.home() / "Programming/Python/machine-learning-exercises/"
                  "uci-ml-repository/diabetes-in-130-US-hospitals"
)
df = pd.read_csv(work_dir / "data/df_encoded.csv")

X = df.drop("readmitted", axis=1)
y = df["readmitted"].copy()

rus = RandomUnderSampler(sampling_strategy="majority", random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)

rf_clf = RandomForestClassifier(
    n_jobs=-1, max_depth=16, n_estimators=160, random_state=42
)
rf_clf.fit(X_resampled, y_resampled)
std_err = np.std([tree.feature_importances_ for tree in rf_clf.estimators_],
                 axis=0)

sorted_features_df = pd.DataFrame(
    {"col_name": rf_clf.feature_importances_, "std_err": std_err},
    index=X.columns).sort_values(by="col_name", ascending=False)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
fig, axes = plt.subplots()
axes.bar(
    sorted_features_df.index,
    sorted_features_df["col_name"],
    yerr=sorted_features_df["std_err"],
    color=colors,
)
plt.setp(axes.get_xticklabels(), ha="right",
         rotation_mode="anchor", rotation=45)
axes.set_title("Feature importances")
plt.savefig("plots/feature_importances.png")
