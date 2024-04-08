#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline

work_dir = Path.home() / 'Programming/Python/machine-learning-exercises/'\
                         'uci-ml-repository/diabetes-in-130-US-hospitals'

# loading data into data frame
df = pd.read_csv(work_dir / 'data/diabetic_data.csv',
                 na_values='?', low_memory=False)
# obj_cols = df.select_dtypes('object').columns
# df[obj_cols] = df[obj_cols].astype('object')

# removing NaN or null values
df = df.loc[df.race.notnull()]
df = df.loc[df.gender != 'Unknown/Invalid', :]

# dropping categories due to high presence of NaN / null values
df.drop(['weight', 'medical_specialty', 'payer_code', 'encounter_id'],
        axis=1, inplace=True)

# converting null/unmapped values in `admission_type_id`,
# `discharge_disposition_id` and `admission_source_id` to NaN.
df['admission_type_id'] = df['admission_type_id'].replace([5, 6, 8], np.nan)
df['discharge_disposition_id'] = df['discharge_disposition_id'].\
    replace([18, 25, 26], np.nan)
df['admission_source_id'] = df['admission_source_id'].\
    replace([9, 15, 17, 20, 21], np.nan)
id_list = ['admission_type_id', 'discharge_disposition_id',
           'admission_source_id']
df[id_list] = df[id_list].astype('object')

# consolidating all variations of `Expired at...` and removing them
df = df[~df['discharge_disposition_id'].isin([11, 19, 20, 21])]

# creating new column called `service_use`
df['service_use'] = df['number_outpatient'] + df['number_emergency'] +\
      df['number_inpatient']
df.drop(['number_outpatient', 'number_emergency', 'number_inpatient'],
        axis=1, inplace=True)

# dropping all duplicate data
df.drop_duplicates(['patient_nbr'], inplace=True)
df.drop(['patient_nbr'], axis=1, inplace=True)

# dropping `citoglipton` and `examide` for lack of discriminatory information
df = df.drop(['citoglipton', 'examide'], axis=1)

# change np.nan to 'None' in 'max_glu_serum' column
df[['max_glu_serum', 'A1Cresult']] = \
    df[['max_glu_serum', 'A1Cresult']].astype('object')
df[['max_glu_serum', 'A1Cresult']] = \
    df[['max_glu_serum', 'A1Cresult']].fillna('None')

# dropping all rows with NaNs
df.dropna(how='any', inplace=True)

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
    # print(f'mask[:10]: {mask[:10]}')
    # new_col_name = diag_name+'_'+str(index)+'_col'
    df.loc[mask, 'primary_diag'] = \
        np.array([diag_name for _ in range(mask.sum())], dtype='object')
    # print(f'new df.head():\n{df.loc[:, new_col_name].head(20)}')

# other diagnosis
diab_others_list = ['Diabetes', 'Others', 'Others']
char_list = ['250.', 'E', 'V']

for diag_name, char in zip(diab_others_list, char_list):
    new_col_name = diag_name+'_'+str(index)+'_col'
    mask = df[diag_col].str.contains(char)
    # df[new_col_name] = mask.astype('object')
    df.loc[mask, 'primary_diag'] = \
        np.array([diag_name for _ in range(mask.sum())])

# dropping all three diagnosis columns
df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

# adding `diabetes` and `others` categories to 'diagnosis_list'
# diag_list = [col+'_1'+'_col' for col in
#              diagnosis_names + ['Diabetes', 'Others']]
df['primary_diag'] = df['primary_diag'].fillna('Others')
df['primary_diag'] = df['primary_diag'].astype('object')
# df.drop(diag_list, axis=1, inplace=True)

# non-readmitted cases, NO and >30 -> 0; readmitted cases, <30 -> 1
df['readmitted'] = df['readmitted'].replace({'<30': '1'})
df['readmitted'] = df['readmitted'].replace({'>30': '0'})
df['readmitted'] = df['readmitted'].replace({'NO': '0'})
df['readmitted'] = df['readmitted'].astype('object')

# resetting index
df.reset_index(inplace=True, drop=True)

# category lists
one_category_list = ['glimepiride-pioglitazone',
                     'metformin-rosiglitazone']  # No
two_category_list = ['acetohexamide', 'tolbutamide',
                     'troglitazone', 'tolazamide',
                     'glipizide-metformin',
                     'metformin-pioglitazone']  # 'No', 'Steady'
four_category_list = ['metformin', 'repaglinide',
                      'nateglinide', 'chlorpropamide',
                      'glimepiride', 'glipizide',
                      'glyburide', 'pioglitazone',
                      'rosiglitazone', 'glyburide-metformin',
                      'insulin', 'miglitol']  # 'No', 'Steady', 'Up', 'Down'
race_list = ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic']
gender_list = ['Female', 'Male']
age_list = ['[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)',
            '[70-80)', '[80-90)', '[90-100)', '[0-10)']
primary_diag_list = ['Others', 'Neoplasms', 'Circulatory',
                     'Diabetes', 'Respiratory',
                     'Injury', 'Genitourinary',
                     'Musculoskeletal', 'Digestive']

# category features are coded to numeric values
df.drop(one_category_list, axis=1, inplace=True)
for cat in two_category_list:
    df[cat] = df[cat].replace({key: val for key, val in zip(['No', 'Steady'],
                                                            ['0', '1'])})
for cat in four_category_list:
    df[cat] = df[cat].replace({key: val for key, val in
                               zip(['Down', 'No', 'Steady', 'Up'],
                                   ['0', '1', '1', '2'])})
df[four_category_list] = df[four_category_list].astype('object')
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
df['race'] = df['race'].\
    replace({key: val for key, val in
             zip(race_list,
                 [str(f) for f in range(5)])})
df['gender'] = df['gender'].\
    replace({key: val for key, val in
             zip(gender_list, ['0', '1'])})
df['age'] = df['age'].\
    replace({key: val for key, val in
             zip(age_list,
                 [str(f) for f in
                  list(range(15, 96, 10)) + [5]])})
df['primary_diag'] = df['primary_diag'].\
    replace({key: val for key, val in
             zip(primary_diag_list, [str(f) for f in list(range(9))])})

change_cat_list = ['max_glu_serum', 'A1Cresult', 'acarbose', 'change',
                   'diabetesMed', 'race', 'gender', 'age', 'primary_diag']
df[change_cat_list] = df[change_cat_list].astype('object')

# listing category features as 'object'
cat_list = ['max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
            'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
            'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'metformin-pioglitazone', 'change',
            'diabetesMed', 'readmitted', 'primary_diag',
            'admission_type_id', 'discharge_disposition_id',
            'admission_source_id']
df[cat_list] = df[cat_list].astype('object')

# are there null values?
# print(f'Null values: \n{df.isnull().any()}\n')
print(f'Dataframe shape:{df.shape}')
print(f'Readmitted cases count: {df.readmitted.value_counts()}')

# standardize numeric data and generate one-hot encoded data features
num_attrib = df.select_dtypes('int').columns
cat_attrib = df.select_dtypes('object').columns

num_pipeline = make_pipeline(StandardScaler())
cat_pipeline = make_pipeline(OneHotEncoder())

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_attrib),
    ('cat', cat_pipeline, cat_attrib)
    ])

arr_prepared = preprocessing.fit_transform(df)
prepared_columns = preprocessing.get_feature_names_out()
df_prepared = pd.DataFrame(arr_prepared, columns=prepared_columns)

# saving dataframe to CSV file
df.to_csv('data/df_encoded.csv', index=False)
print('Done!')
