#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

home = os.environ['HOME']

work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/kaggle-notebooks/diabetes-in-130-US-hospitals/'
data_file = work_dir / 'data/diabetic_data.csv'

df = pd.read_csv(data_file, na_values='?', low_memory=False)
obj_cols = df.select_dtypes('object').columns
df[obj_cols] = df[obj_cols].astype('category')

gender_sorted_df = df['gender'].value_counts(dropna=False)
gender_sorted_df = gender_sorted_df/gender_sorted_df.sum()*100
ax = gender_sorted_df.plot(kind='bar', color=plt.cm.Paired.colors, figsize=(8, 5), edgecolor='k')
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=12)
ax.set_title('Percentage of patients for each gender category', fontsize=16)
ax.set_ylabel('Percent (%)', fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.savefig('plots/gender_sorted.png', dpi=288, bbox_inches='tight')
