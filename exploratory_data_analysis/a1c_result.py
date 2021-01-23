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

a1c_result_df = df['A1Cresult'].value_counts(dropna=False)/df.shape[0]*100
ax = a1c_result_df.plot(kind='bar', color=plt.cm.Paired.colors, figsize=(8, 5), edgecolor='k')
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
ax.set_ylabel('Percent (%)', fontsize=14)
ax.set_title('Percentage of HbA1c measurements per category', fontsize=16)
plt.savefig('plots/a1c_result.png', dpi=288, bbox_inches='tight')
