#! /usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import PurePath


def conf_mx_heat_plot(conf_mx, model_dir):
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx/row_sums
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.heatmap(norm_conf_mx, cmap=plt.cm.coolwarm, ax=ax, square=True, vmin=0, vmax=1,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                annot=True, fmt='.3f', color='w', annot_kws={'fontsize': 10})
    plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor", rotation=0, fontsize=9)
    plt.setp(ax.get_yticklabels(), ha="center", rotation_mode="anchor", rotation=90, fontsize=9)
    plt.tick_params(which='both', bottom=False, left=False)
    ax.set_title('Confusion matrix heat map', fontsize=11)
    ax.set_xlabel('Predicted values', fontsize=9)
    ax.set_ylabel('Actual values', fontsize=9)
    fig.tight_layout()
    plt.savefig(PurePath(model_dir) / 'confusion-matrix-heatmap.png', dpi=288)
