#! /usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import PurePath


def conf_mx_heat_plot(conf_mx, model_name, plot_dir):
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx/row_sums
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.heatmap(norm_conf_mx, cmap=plt.cm.coolwarm, ax=ax, square=True, vmin=0, vmax=1,
                xticklabels=['Not readmitted', 'Readmitted'],
                yticklabels=['Not readmitted', 'Readmitted'],
                annot=True, fmt='.3f', color='w', annot_kws={'fontsize': 10})
    plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor", rotation=0, fontsize=9)
    plt.setp(ax.get_yticklabels(), ha="center", rotation_mode="anchor", rotation=90, fontsize=9)
    plt.tick_params(which='both', bottom=False, left=False)
    ax.set_title(f'Confusion matrix heatmap for {model_name.lower()}', fontsize=10)
    ax.set_xlabel('Predicted values', fontsize=9)
    ax.set_ylabel('Actual values', fontsize=9)
    fig.tight_layout()
    plot_file = 'cm_heatmap_'+('_'.join(model_name.split(' ')).lower())+'.png'
    plt.savefig(PurePath(plot_dir) / plot_file, dpi=288)
