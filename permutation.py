#! /usr/bin/env python

import pandas as pd
import json
from pathlib import Path
from time import time
from datetime import timedelta
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

plt.style.use("style_files/boxplot-style.mplstyle")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

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

print(f'Number of rows: {df.shape[0]}, number of columns: {df.shape[1]}.')
t0 = time()

# load nested parameters from JSON file
with open('params.json', 'r') as f:
    model_params = json.load(f)

rus = RandomUnderSampler(sampling_strategy='majority', random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)

gbc = GradientBoostingClassifier(
            learning_rate=model_params['params_gb']['learning_rate'],
            random_state=model_params['params_gb']['random_state'],
        )

gbc_clf = gbc.fit(X_resampled, y_resampled)
cols = gbc_clf.feature_names_in_
r = permutation_importance(gbc_clf, X_resampled, y_resampled,
                           scoring='balanced_accuracy', n_jobs=-1,
                           random_state=42, n_repeats=25,)

sorted_results_dict = {}
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        sorted_results_dict[cols[i]] = r.importances[i]

# boxplot
num = 12
first_elements_from_dict = {key: value for i, (key, value) in
                            enumerate(sorted_results_dict.items())
                            if i <= num}
fig, axes = plt.subplots(nrows=1, ncols=1)
bplot = axes.boxplot(first_elements_from_dict.values(),
                     labels=first_elements_from_dict.keys(),
                     patch_artist=True)
axes.set_xlabel('Features')
axes.set_ylabel('Decrease in accuracy score')
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('0.2')
    patch.set_alpha(0.95)

plt.setp(axes.get_xticklabels(), ha="right",
         rotation_mode="anchor", rotation=45)
axes.set_title('Permutation importances of the top {} features'.format(num))
plt.savefig(work_dir / 'plots/permutation_importances.png')

results_df = pd.DataFrame(dict(list(r.items())[:2]), index=cols)
results_df.sort_values(by='importances_mean', ascending=False, inplace=True)
results_df.to_csv(work_dir / 'data/perm_imp.csv')

seconds_rounded = round(time() - t0)
time_elapsed_in_secs = str(timedelta(seconds=seconds_rounded)).split(':')
formatted_time = '{}h {}m {}s'.format(*time_elapsed_in_secs)
print(f'Time elapsed: {formatted_time}.')
