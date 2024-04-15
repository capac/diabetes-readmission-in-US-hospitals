#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path

work_dir = Path.home() / 'Programming/Python/machine-learning-exercises/'\
                         'uci-ml-repository/diabetes-in-130-US-hospitals'

# loading data into data frame
df = pd.read_csv(work_dir / 'data/diabetic_data.csv',
                 na_values='?', low_memory=False)

# the following is necessary because DictVectorizer will only do a
# binary one-hot encoding when feature values are of type string.
obj_cols = df.select_dtypes('object').columns
df[obj_cols] = df[obj_cols].astype('string')

# removing NaN or null values from race and gender
df = df.loc[df.race.notnull()]
df = df.loc[df.gender != 'Unknown/Invalid', :]

# dropping categories due to high presence of NaN / null values
df.drop(['weight', 'medical_specialty', 'payer_code', 'encounter_id'],
        axis=1, inplace=True)

# converting null/unmapped values in 'admission_type_id',
# 'discharge_disposition_id' and 'admission_source_id' to NaN.
df['admission_type_id'] = df['admission_type_id'].replace([5, 6, 8], np.nan)
df['discharge_disposition_id'] = df['discharge_disposition_id'].\
    replace([18, 25, 26], np.nan)
df['admission_source_id'] = df['admission_source_id'].\
    replace([9, 15, 17, 20, 21], np.nan)

# consolidating all variations of `Expired at...` and removing them
df = df[~df['discharge_disposition_id'].isin([11, 19, 20, 21])]

# converting 'admission_type_id', 'discharge_disposition_id'
# and 'admission_source_id' from int64 to string
id_list = ['admission_type_id', 'discharge_disposition_id',
           'admission_source_id']
df[id_list] = df[id_list].astype('string')

# creating new column called `service_use`
df['service_use'] = df['number_outpatient'] + df['number_emergency'] +\
      df['number_inpatient']

# removing 'number_outpatient', 'number_emergency', 'number_inpatient' columns
df.drop(['number_outpatient', 'number_emergency', 'number_inpatient'],
        axis=1, inplace=True)

# returning current shape of dataframe
print(f'Dataframe shape:{df.shape}')

# dropping all duplicate patient data
df.drop_duplicates(['patient_nbr'], inplace=True)
df.drop(['patient_nbr'], axis=1, inplace=True)

# returning shape of dataframe after removing patient duplicates
print(f'Dataframe shape:{df.shape}')

# dropping `citoglipton` and `examide` for lack of discriminatory information
df = df.drop(['citoglipton', 'examide'], axis=1)

# change np.nan to 'None' in 'max_glu_serum' and 'A1Cresult' columns
df[['max_glu_serum', 'A1Cresult']] = \
    df[['max_glu_serum', 'A1Cresult']].fillna('None')

# dropping all rows with NaNs
df.dropna(how='any', inplace=True)

# dataframe aftering removing all remaining rows with NAs
print(f'Dataframe shape:{df.shape}')

# grouping primary diagnosis values into group categories
circulatory_list = [str(f) for f in list(range(390, 460)) + [785]]
respiratory_list = [str(f) for f in list(range(460, 520)) + [786]]
digestive_list = [str(f) for f in list(range(520, 580)) + [787]]
injury_list = [str(f) for f in list(range(800, 1000))]
musculoskeletal_list = [str(f) for f in list(range(710, 740))]
genitourinary_list = [str(f) for f in list(range(580, 630)) + [788]]
neoplasms_list = [str(f) for f in list(range(140, 240))]

diagnosis_list = [circulatory_list, respiratory_list, digestive_list,
                  injury_list, musculoskeletal_list, genitourinary_list,
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

# non-readmitted cases, NO and >30 -> 0; readmitted cases, <30 -> 1
df['readmitted'] = df['readmitted'].replace({'<30': '1'})
df['readmitted'] = df['readmitted'].replace({'>30': '0'})
df['readmitted'] = df['readmitted'].replace({'NO': '0'})

# resetting index
df.reset_index(inplace=True, drop=True)

# one category lists
one_category_list = ['glimepiride-pioglitazone',
                     'metformin-rosiglitazone']  # No
df.drop(one_category_list, axis=1, inplace=True)

# two category list
two_category_list = ['acetohexamide', 'tolbutamide',
                     'troglitazone', 'tolazamide',
                     'glipizide-metformin',
                     'metformin-pioglitazone']  # 'No', 'Steady'
for cat in two_category_list:
    df[cat] = df[cat].replace({key: val for key, val in zip(['No', 'Steady'],
                                                            ['0', '1'])})
df[two_category_list] = df[two_category_list].astype('string')

# four category list
four_category_list = ['metformin', 'repaglinide',
                      'nateglinide', 'chlorpropamide',
                      'glimepiride', 'glipizide',
                      'glyburide', 'pioglitazone',
                      'rosiglitazone', 'glyburide-metformin',
                      'insulin', 'miglitol']  # 'No', 'Steady', 'Up', 'Down'
for cat in four_category_list:
    df[cat] = df[cat].replace({key: val for key, val in
                               zip(['Down', 'No', 'Steady', 'Up'],
                                   ['0', '1', '1', '2'])})
df[four_category_list] = df[four_category_list].astype('string')

# race list
race_list = ['Caucasian', 'AfricanAmerican',
             'Other', 'Asian', 'Hispanic']
df['race'] = df['race'].\
    replace({key: val for key, val in
             zip(race_list,
                 [str(f) for f in range(5)])})
df['race'] = df['race'].astype('string')

# gender list
gender_list = ['Female', 'Male']
df['gender'] = df['gender'].\
    replace({key: val for key, val in
             zip(gender_list, ['0', '1'])})
df['gender'] = df['gender'].astype('string')

# age list
age_list = ['[10-20)', '[20-30)', '[30-40)', '[40-50)',
            '[50-60)', '[60-70)', '[70-80)', '[80-90)',
            '[90-100)', '[0-10)']
df['age'] = df['age'].\
    replace({key: val for key, val in
             zip(age_list,
                 [str(f) for f in
                  list(range(15, 96, 10)) + [5]])})
df['age'] = df['age'].astype('string')

# primary diagnosis list
primary_diag_list = ['Others', 'Neoplasms', 'Circulatory',
                     'Diabetes', 'Respiratory',
                     'Injury', 'Genitourinary',
                     'Musculoskeletal', 'Digestive']
df['primary_diag'] = df['primary_diag'].\
    replace({key: val for key, val in
             zip(primary_diag_list, [str(f) for f in list(range(9))])})
df['primary_diag'] = df['primary_diag'].astype('string')


df['max_glu_serum'] = df['max_glu_serum'].\
    replace({key: val for key, val in
             zip(['None', 'Norm', '>200', '>300'],
                 ['0', '1', '2', '2'])})
df['A1Cresult'] = df['A1Cresult'].\
    replace({key: val for key, val in
             zip(['None', 'Norm', '>7', '>8'],
                 ['0', '1', '2', '2'])})
df['acarbose'] = df['acarbose'].\
    replace({key: val for key, val in
             zip(['No', 'Steady', 'Up'],
                 ['0', '1', '2'])})
df['change'] = df['change'].\
    replace({key: val for key, val in
             zip(['No', 'Ch'], ['0', '1'])})
df['diabetesMed'] = df['diabetesMed'].\
    replace({key: val for key, val in
             zip(['No', 'Yes'], ['0', '1'])})
cat_list = ['max_glu_serum', 'A1Cresult', 'acarbose',
            'change', 'diabetesMed']
df[cat_list] = df[cat_list].astype('string')

# final dataframe shape
print(f'Dataframe shape:{df.shape}')
neg, pos = np.bincount(df.readmitted)
total = neg + pos
print(f'Examples:\n   Total: {total}\n'
      f'Negative: {neg} ({100 * neg / total:.2f}% of total)\n'
      f'Positive: {pos} ({100 * pos / total:.2f}% of total)\n')

# saving dataframe to CSV file
df.to_csv('data/df_encoded.csv', index=False)
print('Done!')
