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
            'num_medications', 'number_diagnoses', 'service_use']

corrs = df[num_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corrs, cmap='RdBu_r', fmt='.3f', annot=True, annot_kws={'fontsize': 10})
plt.title('Correlation plot of numeric features', fontsize=16)
plt.savefig(work_dir / 'plots/corr_plot.png', dpi=288, bbox_inches='tight')
