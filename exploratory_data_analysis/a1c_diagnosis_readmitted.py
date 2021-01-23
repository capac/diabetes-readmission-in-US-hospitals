#! /usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

home = os.environ['HOME']

work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals/'
data_file = work_dir / 'data/diabetic_data.csv'

df = pd.read_csv(data_file, na_values='?', low_memory=False)
obj_cols = df.select_dtypes('object').columns
df[obj_cols] = df[obj_cols].astype('category')

# A1Cresult
a1c_result_df = df['A1Cresult'].value_counts(dropna=False, normalize=True)*100
a1c_result_df = a1c_result_df.reset_index().rename(columns={'index': 'Feature', 'A1Cresult': 'Percentage'})

# readmitted
readmitted_df = df['readmitted'].value_counts(dropna=False, normalize=True)*100
readmitted_df = readmitted_df.reset_index().rename(columns={'index': 'Feature', 'A1Cresult': 'Percentage'})

# primary diagnosis
df = df.loc[df.race.notnull()]
df = df.loc[df.gender != 'Unknown/Invalid', :]

df.drop(['weight', 'medical_specialty', 'payer_code', 'encounter_id'], axis=1, inplace=True)

df['admission_type_id'] = df['admission_type_id'].replace([5, 6, 8], np.nan)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace([18, 25, 26], np.nan)
df['admission_source_id'] = df['admission_source_id'].replace([9, 15, 17, 20, 21], np.nan)
id_list = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
df[id_list] = df[id_list].astype('category')

df = df[~df['discharge_disposition_id'].isin([11, 19, 20, 21])]

df['service_use'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
df.drop(['number_outpatient', 'number_emergency', 'number_inpatient'], axis=1, inplace=True)

df.dropna(how='any', inplace=True)

circulatory_list = [str(f) for f in list(range(390, 460)) + [785]]
respiratory_list = [str(f) for f in list(range(460, 520)) + [786]]
digestive_list = [str(f) for f in list(range(520, 580)) + [787]]
injury_list = [str(f) for f in list(range(800, 1000))]
musculoskeletal_list = [str(f) for f in list(range(710, 740))]
genitourinary_list = [str(f) for f in list(range(580, 630)) + [788]]
neoplasms_list = [str(f) for f in list(range(140, 240))]

diagnosis_list = [circulatory_list, respiratory_list, digestive_list, injury_list,
                  musculoskeletal_list, genitourinary_list, neoplasms_list]
diagnosis_names = ['Circulatory', 'Respiratory', 'Digestive', 'Injury',
                   'Musculoskeletal', 'Genitourinary', 'Neoplasms']

for diag_name, diag_list in zip(diagnosis_names, diagnosis_list):
    df.loc[:, diag_name+'_col'] = np.array([np.nan for i in range(df.shape[0])])
    filter_ = df['diag_1'].isin(diag_list)
    df.loc[filter_, diag_name+'_col'] = np.array([diag_name for i in range(filter_.sum())])

diab_others_list = ['Diabetes', 'Others', 'Others']
char_list = ['250.', 'E', 'V']

for diag_name, char in zip(diab_others_list, char_list):
    df.loc[:, diag_name+'_col'] = np.array([np.nan for _ in range(df.shape[0])])
    filter_ = df['diag_1'].str.contains(char)
    df.loc[filter_, diag_name+'_col'] = np.array([diag_name for i in range(filter_.sum())])

tmp_diag_list = [col+'_col' for col in diagnosis_names + ['Diabetes', 'Others']]
df['primary_diag'] = df[tmp_diag_list].fillna(axis=1, method='bfill').iloc[:, 0]
df['primary_diag'].fillna(value='Others', inplace=True)
df['primary_diag'] = df['primary_diag'].astype('category')
df.drop(tmp_diag_list, axis=1, inplace=True)

diagnosis_df = df['primary_diag'].value_counts(dropna=False)
diagnosis_df.sort_values(ascending=False, inplace=True)
diagnosis_df = (diagnosis_df/diagnosis_df.sum()*100).apply('{:.2f}'.format).astype('float')
diagnosis_df = diagnosis_df.reset_index().rename(columns={'index': 'Primary diagnosis', 'primary_diag': 'Percentage'})

order_list = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury',
              'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Others']
diagnosis_df = diagnosis_df.set_index('Primary diagnosis').loc[order_list]
diagnosis_df = diagnosis_df.reset_index().rename(columns={'Primary diagnosis': 'Feature'})

df_list = [a1c_result_df, diagnosis_df, readmitted_df]
title_list = ['Percentage of HbA1c measurements',
              'Percentage of patients per primary diagnosis',
              'Percentage of readmissions']
wd_list = [0.6, 0.6, 0.5]

# plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for df, ax, title, wd in zip(df_list, axes, title_list, wd_list):
    ax.bar(df.iloc[:, 0], df.iloc[:, 1], color=plt.cm.Paired.colors, edgecolor='k', width=wd)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.set_ylabel('Percent (%)', fontsize=10)
    ax.set_title(title, fontsize=12)
plt.savefig('plots/a1c_diagnosis_readmitted.png', dpi=288, bbox_inches='tight')
