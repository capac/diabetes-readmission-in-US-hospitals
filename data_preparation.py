#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path
import os
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler


home = os.environ['HOME']

work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/kaggle-notebooks/diabetes-in-130-US-hospitals/'

# loading data into data frame
df = pd.read_csv(work_dir / 'data/diabetic_data.csv', na_values='?', low_memory=False)
obj_cols = df.select_dtypes('object').columns
df[obj_cols] = df[obj_cols].astype('category')

# removing NaN or null values
df = df.loc[df.race.notnull()]
df = df.loc[df.gender != 'Unknown/Invalid', :]

# dropping categories due to high presence of NaN / null values
df.drop(['weight', 'medical_specialty', 'payer_code', 'encounter_id'], axis=1, inplace=True)

# converting null/unmapped values in `admission_type_id`, `discharge_disposition_id` and `admission_source_id` to NaN.
df['admission_type_id'] = df['admission_type_id'].replace([5, 6, 8], np.nan)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace([18, 25, 26], np.nan)
df['admission_source_id'] = df['admission_source_id'].replace([9, 15, 17, 20, 21], np.nan)
id_list = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
df[id_list] = df[id_list].astype('category')

# consolidating all variations of `Expired at...` and removing them
df = df[~df['discharge_disposition_id'].isin([11, 19, 20, 21])]

# creating new column called `service_use`
df['service_use'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
df.drop(['number_outpatient', 'number_emergency', 'number_inpatient'], axis=1, inplace=True)

# dropping all rows with NaNs
df.dropna(how='any', inplace=True)

# dropping all duplicate data
df.drop_duplicates(['patient_nbr'], inplace=True)
df.drop(['patient_nbr'], axis=1, inplace=True)

# dropping `citoglipton` and `examide` for lack of discriminatory information
df = df.drop(['citoglipton', 'examide'], axis=1)

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

for diag_name, diag_list in zip(diagnosis_names, diagnosis_list):
    df.loc[:, diag_name+'_'+str(index)+'_col'] = np.array([np.nan for i in range(df.shape[0])])
    filter_ = df[diag_col].isin(diag_list)
    df.loc[filter_, diag_name+'_'+str(index)+'_col'] = np.array([diag_name for i in range(filter_.sum())])

# other diagnosis
diab_others_list = ['Diabetes', 'Others', 'Others']
char_list = ['250.', 'E', 'V']

for diag_name, char in zip(diab_others_list, char_list):
    df.loc[:, diag_name+'_'+str(index)+'_col'] = np.array([np.nan for i in range(df.shape[0])])
    filter_ = df[diag_col].str.contains(char)
    df.loc[filter_, diag_name+'_'+str(index)+'_col'] = np.array([diag_name for i in range(filter_.sum())])

# dropping all three diagnosis columns
df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

# adding `diabetes` and `others` categories
diag_list = [col+'_1'+'_col' for col in diagnosis_names + ['Diabetes', 'Others']]
df['primary_diag'] = df[diag_list].fillna(axis=1, method='bfill').iloc[:, 0]
df['primary_diag'].fillna(value='Others', inplace=True)
df['primary_diag'] = df['primary_diag'].astype('category')
df.drop(diag_list, axis=1, inplace=True)

# non-readmitted cases, NO and >30 -> 0; readmitted cases, <30 -> 1
df['readmitted'].replace('<30', '1', inplace=True)
df['readmitted'].replace(['>30', 'NO'], '0', inplace=True)
df['readmitted'] = df['readmitted'].astype('int64')

# resetting index
df.reset_index(inplace=True, drop=True)

# category lists
one_category_list = ['glimepiride-pioglitazone', 'metformin-rosiglitazone']  # No
two_category_list = ['acetohexamide', 'tolbutamide', 'troglitazone', 'tolazamide',
                     'glipizide-metformin', 'metformin-pioglitazone']  # 'No', 'Steady'
four_category_list = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                      'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
                      'rosiglitazone', 'miglitol', 'insulin', 'glyburide-metformin']  # 'No', 'Steady', 'Up', 'Down'
race_list = ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic']
gender_list = ['Female', 'Male']
age_list = ['[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)',
            '[70-80)', '[80-90)', '[90-100)', '[0-10)']
primary_diag_list = ['Others', 'Neoplasms', 'Circulatory', 'Diabetes', 'Respiratory',
                     'Injury', 'Genitourinary', 'Musculoskeletal', 'Digestive']

# category features are coded to numeric values
df.drop(one_category_list, axis=1, inplace=True)
for cat in two_category_list:
    df[cat] = df[cat].replace(['No', 'Steady'], [0, 1])
for cat in four_category_list:
    df[cat] = df[cat].replace(['Down', 'No', 'Steady', 'Up'], [0, 1, 1, 2])
df['max_glu_serum'] = df['max_glu_serum'].replace(['None', 'Norm', '>200', '>300'], [0, 1, 2, 2])
df['A1Cresult'] = df['A1Cresult'].replace(['None', 'Norm', '>7', '>8'], [0, 1, 2, 2])
df['acarbose'] = df['acarbose'].replace(['No', 'Steady', 'Up'], [0, 1, 2])
df['change'] = df['change'].replace(['No', 'Ch'], [0, 1])
df['diabetesMed'] = df['diabetesMed'].replace(['No', 'Yes'], [0, 1])
df['race'] = df['race'].replace(race_list, [0, 1, 2, 3, 4])
df['gender'] = df['gender'].replace(gender_list, [0, 1])
df['age'] = df['age'].replace(age_list, [15, 25, 35, 45, 55, 65, 75, 85, 95, 5])
df['primary_diag'] = df['primary_diag'].replace(primary_diag_list, [0, 1, 2, 3, 4, 5, 6, 7, 8])
df[['race', 'gender', 'age']] = df[['race', 'gender', 'age']].astype('category')

# making sure caetgory features are such
cat_list = ['max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
            'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
            'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted', 'primary_diag',
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
df[cat_list] = df[cat_list].astype('category')

# are there null values?
# print(f'Null values: \n{df.isnull().any()}')

# SMOTE: Synthetic Minority Over-sampling Technique
# When dealing with mixed data type such as continuous and categorical features,
# none of the presented methods (apart of the class RandomOverSampler) can deal
# with the categorical features.
# https://imbalanced-learn.org/stable/over_sampling.html#smote-variants
input_features = list(set(df.columns) - set(['readmitted']))
X, y = df[input_features], df['readmitted']
sm = RandomOverSampler(sampling_strategy='minority')
X_new, y_new = sm.fit_resample(X, y)
df = pd.concat([X_new, y_new], axis=1)

# are there null values?
# print(f'Null values after SMOTE: \n{df.isnull().any()}')

# standardize numeric data
scaler_encoder = StandardScaler()
num_cols = df.select_dtypes('int64').columns
df_num = df[num_cols]
X_num = scaler_encoder.fit_transform(df_num)
df_num = pd.DataFrame(X_num, index=df_num.index, columns=num_cols)
df_cat = df.drop(num_cols, axis=1)
df = pd.concat([df_num, df_cat], axis=1)

# saving dataframe to CSV file
df.to_csv('data/df_encoded.csv', index=False)
print('Done!')
