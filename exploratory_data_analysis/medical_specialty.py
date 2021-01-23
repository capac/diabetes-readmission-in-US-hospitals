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

medical_specialty_df = df['medical_specialty'].value_counts(dropna=False, normalize=True)*100
medical_specialty_df = medical_specialty_df.iloc[0:20]
medical_specialty_df = medical_specialty_df.reset_index().rename(columns={'index': 'Feature',
                                                                 'medical_specialty': 'Medical Specialty'})
medical_specialty_df['Feature'] = medical_specialty_df['Feature'].astype('category')
medical_specialty_df['Feature'] = medical_specialty_df['Feature'].cat.add_categories('Missing').fillna('Missing')

fig, axes = plt.subplots(figsize=(16, 6))
axes.bar(medical_specialty_df['Feature'], medical_specialty_df['Medical Specialty'],
         color=plt.cm.Paired.colors, edgecolor='k')
plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=14)
plt.setp(axes.get_yticklabels(), fontsize=14)
axes.set_ylabel('Percent (%)', fontsize=14)
axes.set_title('Percentage breakdown of first twenty medical specialties', fontsize=16)
plt.savefig('plots/medical_specialty.png', dpi=288, bbox_inches='tight')
