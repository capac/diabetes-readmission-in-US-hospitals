#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("style_files/lineplot-style.mplstyle")

work_dir = Path.home() / 'Programming/Python/machine-learning-exercises/'\
                         'uci-ml-repository/diabetes-in-130-US-hospitals'
df = pd.read_csv(work_dir / 'data/diabetic_data.csv')

num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses']

corrs = df[num_cols].corr(numeric_only=True)

fig, axes = plt.subplots(figsize=(10, 6))
sns.heatmap(corrs, cmap='RdBu_r', fmt='1.3f', annot=True,
            annot_kws={'fontsize': 10}, ax=axes)
plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor",
         rotation=45, fontsize=10)
plt.setp(axes.get_yticklabels(), ha="right", rotation_mode="anchor",
         rotation=0, fontsize=10)
cbar = axes.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)
plt.title('Correlation plot of numeric features', fontsize=12)
fig.tight_layout()
plt.savefig(work_dir / 'plots/corr_plot.png', dpi=288)
