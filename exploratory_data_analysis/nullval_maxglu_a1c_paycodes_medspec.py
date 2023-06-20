#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path

home = os.environ['HOME']
work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/'\
                        'uci-ml-repository/diabetes-in-130-US-hospitals/'
data_file = work_dir / 'data/diabetic_data.csv'

# bar plot style with directory path
barplot_style = work_dir / 'barplot-style.mplstyle'
plt.style.use(barplot_style)

df = pd.read_csv(data_file, na_values='?', low_memory=False)
obj_cols = df.select_dtypes('object').columns
df[obj_cols] = df[obj_cols].astype('category')

# missing values
missing_values = df.isnull().sum()/df.shape[0]*100
missing_values.sort_values(ascending=False, inplace=True)
missing_df = missing_values.reset_index().\
    rename(columns={'index': 'Feature', 0: 'Percentage'})
missing_df['Percentage'] = missing_df['Percentage'].\
    apply('{:.4f}'.format).astype('float')
missing_df = missing_df.loc[missing_df['Percentage'] != 0]

# max_glu_serum
max_glu_serum_df = df.max_glu_serum.value_counts(dropna=False, normalize=True)*100
max_glu_serum_df = max_glu_serum_df.to_frame().reset_index().\
    rename(columns={'max_glu_serum': 'Feature',
                    'proportion': 'Proportion'})
max_glu_serum_df['Feature'] = max_glu_serum_df['Feature'].\
    cat.add_categories('Missing').fillna('Missing')

# A1C result
A1Cresult_df = df.A1Cresult.value_counts(dropna=False, normalize=True)*100
A1Cresult_df = A1Cresult_df.to_frame().reset_index().\
    rename(columns={'A1Cresult': 'Feature', 'proportion': 'Proportion'})
A1Cresult_df['Feature'] = A1Cresult_df['Feature'].\
    cat.add_categories('Missing').fillna('Missing')

# payer codes
payer_code_sorted_df = df['payer_code'].value_counts(dropna=False,
                                                     normalize=True)*100
payer_code_sorted_df = payer_code_sorted_df.iloc[0:15]
payer_code_sorted_df = payer_code_sorted_df.reset_index().\
    rename(columns={'proportion': 'Proportion', 'payer_code': 'Payer Code'})
payer_code_sorted_df['Payer Code'] = \
    payer_code_sorted_df['Payer Code'].\
    astype('category')
payer_code_sorted_df['Payer Code'] = \
    payer_code_sorted_df['Payer Code'].\
    cat.add_categories('Missing').fillna('Missing')

# medical speciality
medical_specialty_df = df['medical_specialty'].\
    value_counts(dropna=False, normalize=True)*100
medical_specialty_df = medical_specialty_df.iloc[0:20]
medical_specialty_df = medical_specialty_df.reset_index().\
    rename(columns={'proportion': 'Proportion',
                    'medical_specialty': 'Medical Specialty'})
medical_specialty_df['Medical Specialty'] = \
    medical_specialty_df['Medical Specialty'].\
    astype('category')
medical_specialty_df['Medical Specialty'] = \
    medical_specialty_df['Medical Specialty'].\
    cat.add_categories('Missing').fillna('Missing')

# print(f'missing_df: {missing_df}')
# print(f'payer_code_sorted_df: {payer_code_sorted_df}')
# print(f'medical_specialty_df: {medical_specialty_df}')

df_list = [missing_df, max_glu_serum_df, A1Cresult_df,
           medical_specialty_df, payer_code_sorted_df]
title_list = ['Percentage of missing values per feature',
              'Percent breakdown of max_glu_serum results',
              'Percent breakdown of HbA1c results',
              'Percent breakdown of first 20 medical specialties',
              'Percent breakdown of first 15 payer codes']
width_list = [0.6, 0.6, 0.6, 0.6, 0.5]
fontsize_list = [11, 11, 11, 10, 10]

# plot settings
plt.rcParams['ytick.major.pad'] = -10
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# plot
fig = plt.figure(tight_layout=True, figsize=(16, 10))
gs = GridSpec(6, 3, figure=fig)
# main plot
ax0 = fig.add_subplot(gs[1:5, 0])
# secondary plots
ax1 = fig.add_subplot(gs[0:3, 1])
ax2 = fig.add_subplot(gs[0:3, 2])
ax3 = fig.add_subplot(gs[3:6:, 1])
ax4 = fig.add_subplot(gs[3:6:, 2])
axes = [ax0, ax1, ax2, ax3, ax4]

for df, ax, title, wd, ft in zip(df_list, axes, title_list,
                                 width_list, fontsize_list):
    ax.bar(df.iloc[:, 0], df.iloc[:, 1], color=colors, width=wd)
    plt.setp(ax.get_xticklabels(), ha="right", fontsize=ft,
             rotation_mode="anchor", rotation=45)
    plt.setp(ax.get_yticklabels(), fontsize=11)
    ax.set_ylabel('Percent (%)')
    ax.set_title(title, fontsize=12)
plt.savefig(work_dir / 'exploratory_data_analysis/plots/'
            'nullval_maxglu_a1c_paycodes_medspec.png')
