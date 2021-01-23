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

payer_code_sorted_df = df['payer_code'].value_counts(dropna=False, normalize=True)*100
payer_code_sorted_df = payer_code_sorted_df.iloc[0:15]
payer_code_sorted_df = payer_code_sorted_df.reset_index().rename(columns={'index': 'Feature',
                                                                 'payer_code': 'Payer Code'})
payer_code_sorted_df['Feature'] = payer_code_sorted_df['Feature'].astype('category')
payer_code_sorted_df['Feature'] = payer_code_sorted_df['Feature'].cat.add_categories('Missing').fillna('Missing')

fig, axes = plt.subplots(figsize=(16, 6))
axes.bar(payer_code_sorted_df['Feature'], payer_code_sorted_df['Payer Code'], color=plt.cm.Paired.colors, edgecolor='k')
plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=14)
plt.setp(axes.get_yticklabels(), fontsize=14)
axes.set_ylabel('Percent (%)', fontsize=14)
axes.set_title('Percentage breakdown for first 15 payer codes', fontsize=16)
plt.savefig('plots/payer_code.png', dpi=288, bbox_inches='tight')
