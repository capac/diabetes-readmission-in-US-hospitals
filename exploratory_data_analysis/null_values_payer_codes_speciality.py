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

# missing values
missing_values = df.isnull().sum()/df.shape[0]*100
missing_values.sort_values(ascending=True, inplace=True)
missing_df = missing_values.reset_index().rename(columns={'index': 'Feature', 0: 'Percentage'})
missing_df['Percentage'] = missing_df['Percentage'].apply('{:.4f}'.format).astype('float')
missing_df = missing_df.loc[missing_df['Percentage'] != 0].sort_values(by='Percentage', ascending=False)

# payer codes
payer_code_sorted_df = df['payer_code'].value_counts(dropna=False, normalize=True)*100
payer_code_sorted_df = payer_code_sorted_df.iloc[0:15]
payer_code_sorted_df = payer_code_sorted_df.reset_index().rename(columns={'index': 'Feature',
                                                                 'payer_code': 'Payer Code'})
payer_code_sorted_df['Feature'] = payer_code_sorted_df['Feature'].astype('category')
payer_code_sorted_df['Feature'] = payer_code_sorted_df['Feature'].cat.add_categories('Missing').fillna('Missing')

# medical speciality
medical_specialty_df = df['medical_specialty'].value_counts(dropna=False, normalize=True)*100
medical_specialty_df = medical_specialty_df.iloc[0:20]
medical_specialty_df = medical_specialty_df.reset_index().rename(columns={'index': 'Feature',
                                                                 'medical_specialty': 'Medical Specialty'})
medical_specialty_df['Feature'] = medical_specialty_df['Feature'].astype('category')
medical_specialty_df['Feature'] = medical_specialty_df['Feature'].cat.add_categories('Missing').fillna('Missing')

# print(f'missing_df: {type(missing_df)}')
# print(f'payer_code_sorted_df: {type(payer_code_sorted_df)}')
# print(f'medical_specialty_df: {type(medical_specialty_df)}')

df_list = [missing_df, medical_specialty_df, payer_code_sorted_df]
title_list = ['Percentage of missing values per feature',
              'Percentage of missing values for first 20 medical specialties',
              'Percentage of missing values for first 15 payer codes']
wd_list = [0.6, 0.6, 0.5]
fontsize_list = [10, 8, 9]

# plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for df, ax, title, wd, ft in zip(df_list, axes, title_list, wd_list, fontsize_list):
    ax.bar(df.iloc[:, 0], df.iloc[:, 1], color=plt.cm.Paired.colors, edgecolor='k', width=wd)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=ft)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.set_ylabel('Percent (%)', fontsize=10)
    ax.set_title(title, fontsize=10)
plt.savefig('plots/null_values_payer_codes_speciality.png', dpi=288, bbox_inches='tight')
