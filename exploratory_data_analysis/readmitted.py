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

readmitted_df = df['readmitted'].value_counts(dropna=False)
readmitted_df = readmitted_df/readmitted_df.sum()*100
ax = readmitted_df.plot(kind='bar', color=plt.cm.Paired.colors, figsize=(8, 5), edgecolor='k')
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.set_ylabel('Percent (%)', fontsize=14)
ax.set_title('Percentage of patients readmitted', fontsize=16)
plt.savefig('plots/readmitted.png', dpi=288, bbox_inches='tight')
