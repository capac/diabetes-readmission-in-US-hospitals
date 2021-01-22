#! /usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

home = os.environ['HOME']

work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/kaggle-notebooks/diabetes-in-130-US-hospitals/'
data_file = work_dir / 'data/diabetic_data.csv'

df = pd.read_csv(data_file, na_values='?', low_memory=False)
obj_cols = df.select_dtypes('object').columns
df[obj_cols] = df[obj_cols].astype('category')

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
diagnosis_sr = pd.Series(diagnosis_df.Percentage, index=diagnosis_df.index)

ax = diagnosis_sr.plot(kind='bar', color=plt.cm.Paired.colors, figsize=(8, 5), edgecolor='k')
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
ax.set_xlabel(None)
ax.set_ylabel('Percent (%)', fontsize=14)
ax.set_title('Percentage of patients per condition for primary diagnosis', fontsize=16)
plt.savefig('plots/diagnosis.png', dpi=288, bbox_inches='tight')
