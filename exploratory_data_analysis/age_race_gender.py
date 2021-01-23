#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

home = os.environ['HOME']

work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals/'
data_file = work_dir / 'data/diabetic_data.csv'

df = pd.read_csv(data_file, na_values='?', low_memory=False)
obj_cols = df.select_dtypes('object').columns
df[obj_cols] = df[obj_cols].astype('category')

# age
age_df = df['age'].value_counts(dropna=False, normalize=True)*100
age_df = age_df.reset_index().rename(columns={'index': 'Feature', 'age': 'Age'})

# race
race_df = df['race'].value_counts(dropna=False, normalize=True)*100
race_df = race_df.reset_index().rename(columns={'index': 'Feature', 'race': 'Race'})
race_df['Feature'] = race_df['Feature'].astype('category')
race_df['Feature'] = race_df['Feature'].cat.add_categories('Missing').fillna('Missing')

# gender
gender_df = df['gender'].value_counts(dropna=False, normalize=True)*100
gender_df = gender_df.reset_index().rename(columns={'index': 'Feature', 'gender': 'Gender'})

df_list = [age_df, race_df, gender_df]
title_list = ['Percentage for age range',
              'Percentage for race',
              'Percentage for gender']
wd_list = [0.6, 0.6, 0.5]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, title, df, wd in zip(axes, title_list, df_list, wd_list):
    ax.bar(df.iloc[:, 0], df.iloc[:, 1], color=plt.cm.Paired.colors, edgecolor='k', width=wd)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.set_ylabel('Percent (%)', fontsize=10)
    ax.set_title(title, fontsize=12)
plt.savefig('plots/age_race_gender.png', dpi=288, bbox_inches='tight')
