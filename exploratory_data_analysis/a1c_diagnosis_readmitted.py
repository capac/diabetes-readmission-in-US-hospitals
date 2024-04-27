#! /usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

work_dir = Path.home() / 'Programming/Python/machine-learning-exercises/'\
                         'uci-ml-repository/diabetes-in-130-US-hospitals/'
data_file = work_dir / 'data/diabetic_data.csv'

# bar plot style with directory path
barplot_style = work_dir / 'barplot-style.mplstyle'
plt.style.use(barplot_style)

df = pd.read_csv(data_file, na_values='?', low_memory=False)
obj_cols = df.select_dtypes('object').columns
df[obj_cols] = df[obj_cols].astype('string')

# removing NaN or null values from race and gender
df = df.loc[df.race.notnull()]
df = df.loc[df.gender != 'Unknown/Invalid', :]

# dropping categories due to high presence of NaN / null values
df.drop(['weight', 'medical_specialty', 'payer_code', 'encounter_id'],
        axis=1, inplace=True)

# change np.nan to 'None' in 'max_glu_serum' and 'A1Cresult' columns
df[['max_glu_serum', 'A1Cresult']] = \
    df[['max_glu_serum', 'A1Cresult']].fillna('None')

# A1Cresult
a1c_result_df = df['A1Cresult'].value_counts(dropna=False, normalize=True)*100
a1c_result_df = a1c_result_df.reset_index().\
    rename(columns={'A1Cresult': 'Feature', 'proportion': 'Percentage'})

# readmitted
readmitted_df = df['readmitted'].value_counts(dropna=False, normalize=True)*100
readmitted_df = readmitted_df.reset_index().\
    rename(columns={'readmitted': 'Feature', 'proportion': 'Percentage'})
readmitted_df.replace(to_replace={'NO': 'No'}, inplace=True)

# converting null/unmapped values in 'admission_type_id',
# 'discharge_disposition_id' and 'admission_source_id' to NaN.
df['admission_type_id'] = df['admission_type_id'].replace([5, 6, 8], np.nan)
df['discharge_disposition_id'] = df['discharge_disposition_id'].\
    replace([18, 25, 26], np.nan)
df['admission_source_id'] = df['admission_source_id'].\
    replace([9, 15, 17, 20, 21], np.nan)

# converting 'admission_type_id', 'discharge_disposition_id'
# and 'admission_source_id' from int64 to string
id_list = ['admission_type_id', 'discharge_disposition_id',
           'admission_source_id']
df[id_list] = df[id_list].astype('string')

# consolidating all variations of `Expired at...` and removing them
df = df[~df['discharge_disposition_id'].isin([11, 19, 20, 21])]

df['service_use'] = df['number_outpatient'] +\
    df['number_emergency'] + df['number_inpatient']
df.drop(['number_outpatient', 'number_emergency', 'number_inpatient'],
        axis=1, inplace=True)

df.dropna(how='any', inplace=True)

circulatory_list = [str(f) for f in list(range(390, 460)) + [785]]
respiratory_list = [str(f) for f in list(range(460, 520)) + [786]]
digestive_list = [str(f) for f in list(range(520, 580)) + [787]]
injury_list = [str(f) for f in list(range(800, 1000))]
musculoskeletal_list = [str(f) for f in list(range(710, 740))]
genitourinary_list = [str(f) for f in list(range(580, 630)) + [788]]
neoplasms_list = [str(f) for f in list(range(140, 240))]

diagnosis_list = [circulatory_list, respiratory_list,
                  digestive_list, injury_list,
                  musculoskeletal_list, genitourinary_list,
                  neoplasms_list]
diagnosis_names = ['Circulatory', 'Respiratory', 'Digestive', 'Injury',
                   'Musculoskeletal', 'Genitourinary', 'Neoplasms']

# selection on primary diagnosis
diag_col, index = 'diag_1', 1
df['primary_diag'] = np.array(['Others' for _ in range(df.shape[0])],
                              dtype='object')

for diag_name, diag_list in zip(diagnosis_names, diagnosis_list):
    mask = df[diag_col].isin(diag_list)
    df.loc[mask, 'primary_diag'] = \
        np.array([diag_name for _ in range(mask.sum())], dtype='object')

# other diagnosis
diab_others_list = ['Diabetes', 'Others', 'Others']
char_list = ['250.', 'E', 'V']

for diag_name, char in zip(diab_others_list, char_list):
    mask = df[diag_col].str.contains(char)
    df.loc[mask, 'primary_diag'] = \
        np.array([diag_name for _ in range(mask.sum())])

# dropping all three diagnosis columns
df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

diagnosis_df = df['primary_diag'].value_counts(dropna=False,
                                               normalize=True)*100
diagnosis_df.sort_values(ascending=False, inplace=True)

diagnosis_df = diagnosis_df.reset_index().\
    rename(columns={'primary_diag': 'Primary diagnosis',
                    'proportion': 'Percentage'})

order_list = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury',
              'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Others']
diagnosis_df = diagnosis_df.set_index('Primary diagnosis').loc[order_list]
diagnosis_df = diagnosis_df.reset_index().\
    rename(columns={'Primary diagnosis': 'Feature'})

df_list = [a1c_result_df, diagnosis_df, readmitted_df]
title_list = ['Percentage of HbA1c measurements',
              'Percentage of patients per primary diagnosis',
              'Percentage of readmissions']
wd_list = [0.6, 0.6, 0.5]

# colors
plt.rcParams['ytick.major.pad'] = -8
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for df, ax, title, wd in zip(df_list, axes, title_list, wd_list):
    ax.bar(df.iloc[:, 0], df.iloc[:, 1], color=colors, width=wd)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor",
             rotation=45, fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.set_ylabel('Percent (%)', fontsize=10)
    ax.set_title(title, fontsize=12)
plt.savefig(work_dir / 'exploratory_data_analysis/plots/'
            'a1c_diagnosis_readmitted.png')
