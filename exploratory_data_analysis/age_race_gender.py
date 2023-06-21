#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

home = os.environ['HOME']

work_dir = Path.home() / 'Programming/Python/machine-learning-exercises/'\
                         'uci-ml-repository/diabetes-in-130-US-hospitals'
data_file = work_dir / 'data/diabetic_data.csv'

# bar plot style with directory path
barplot_style = work_dir / 'barplot-style.mplstyle'
plt.style.use(barplot_style)

df = pd.read_csv(data_file, na_values='?', low_memory=False)
obj_cols = df.select_dtypes('object').columns
df[obj_cols] = df[obj_cols].astype('category')

# age
age_df = df['age'].value_counts(dropna=False, normalize=True)*100
age_df = age_df.reset_index().rename(columns={'proportion': 'Proportion',
                                              'age': 'Age'})

# race
race_df = df['race'].value_counts(dropna=False, normalize=True)*100
race_df = race_df.reset_index().rename(columns={'proportion': 'Proportion',
                                                'race': 'Race'})
race_df['Race'] = race_df['Race'].astype('category')
race_df['Race'] = race_df['Race'].cat.add_categories('Null').\
    fillna('Null')

# gender
gender_df = df['gender'].value_counts(dropna=False, normalize=True)*100
gender_df = gender_df.reset_index().rename(columns={'proportion': 'Proportion',
                                                    'gender': 'Gender'})

df_list = [age_df, race_df, gender_df]
# for df in df_list:
#     print(df.iloc[:, 0])
#     print(df.iloc[:, 1])
title_list = ['Percentage for age range',
              'Percentage for race',
              'Percentage for gender']
wd_list = [0.6, 0.6, 0.4]

# colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rcParams['ytick.major.pad'] = -8

# plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, title, df, wd in zip(axes, title_list, df_list, wd_list):
    ax.bar(df.iloc[:, 0], df.iloc[:, 1], color=colors, width=wd)
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor", rotation=45)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.set_ylabel('Percent (%)', fontsize=11)
    ax.set_title(title, fontsize=12)
plt.savefig(work_dir / 'exploratory_data_analysis/plots/'
                       'age_race_gender.png')
