#!/usr/bin/env python

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

home = os.environ['HOME']
work_dir = Path(home) / 'Programming/Python/machine-learning-exercises/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/df_encoded.csv')

num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_diagnoses', 'service_use', 'readmitted']

corrs = df[num_cols].corr()

fig, axes = plt.subplots(figsize=(10, 6))
sns.heatmap(corrs, cmap='RdBu_r', fmt='1.3f', annot=True, annot_kws={'fontsize': 10}, ax=axes)
plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=10)
plt.setp(axes.get_yticklabels(), ha="right", rotation_mode="anchor", rotation=0, fontsize=10)
plt.title('Correlation plot of numeric features with readmission', fontsize=16)
fig.tight_layout()
plt.savefig(work_dir / 'plots/corr_plot.png', dpi=288, bbox_inches='tight')
