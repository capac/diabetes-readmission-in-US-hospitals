#! /usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import PurePath


def roc_curve_plot_with_auc(fpr, tpr, model_roc_auc, name, work_dir):
    fig, axes = plt.subplots(figsize=(5, 4))
    axes.plot(fpr, tpr, marker='.', ms=7,
              label='Model: {0:s}, AUC: {1:.4f}'.format(name.lower(), model_roc_auc))
    axes.plot([0, 1], [0, 1], 'r--')
    axes.set_xlim([-0.02, 1.0])
    axes.set_ylim([0.0, 1.02])
    axes.set_xlabel('False Positive Rate', fontsize=9)
    axes.set_ylabel('True Positive Rate', fontsize=9)
    axes.set_title('Receiver operating characteristic for {0:s} model'.format(name.lower()), fontsize=10)
    axes.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    plot_file = '_'.join(name.split(' ')).lower()+'_auc.png'
    plt.savefig(work_dir / 'plots' / plot_file, dpi=288, bbox_inches='tight')


def conf_mx_heat_plot(conf_mx, model_name, plot_dir):
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx/row_sums
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
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
