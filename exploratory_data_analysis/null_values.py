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

missing_values = df.isnull().sum()/df.shape[0]*100
missing_values.sort_values(ascending=True, inplace=True)
missing_df = missing_values.reset_index().rename(columns={'index': 'Feature', 0: 'Percentage'})
missing_df['Percentage'] = missing_df['Percentage'].apply('{:.4f}'.format).astype('float')
missing_df = missing_df.loc[missing_df['Percentage'] != 0].sort_values(by='Percentage', ascending=False)

fig, axes = plt.subplots(figsize=(8, 5))
axes.bar(missing_df['Feature'], missing_df['Percentage'], color=plt.cm.Paired.colors, edgecolor='k')
plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=14)
plt.setp(axes.get_yticklabels(), fontsize=14)
axes.set_ylabel('Percent (%)', fontsize=14)
axes.set_title('Percentage of null values per feature', fontsize=16)
plt.savefig('plots/null_values.png', dpi=288, bbox_inches='tight')
