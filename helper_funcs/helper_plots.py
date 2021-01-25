#! /usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt


def roc_curve_plot_with_auc(rates_dict, work_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    rates_tuples = rates_dict.items()
    for ax, (name, rates_list) in zip(axes, rates_tuples):
        fpr, tpr, model_roc_auc = rates_list
        ax.plot(fpr, tpr, marker='.', ms=6, label='AUC: {:.4f}'.format(model_roc_auc))
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlim([-0.02, 1.0])
        plt.setp(ax.get_xticklabels(), fontsize=10)
        ax.set_ylim([0.0, 1.02])
        plt.setp(ax.get_yticklabels(), fontsize=10)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title('ROC curve for {0:s}'.format(name.lower()), fontsize=11)
        ax.legend(loc='lower right', fontsize=12)
    fig.suptitle('Receiver Operating Characteristics Curves', fontsize=12)
    fig.tight_layout()
    plt.savefig(work_dir / 'plots/auc_plots.png', dpi=288, bbox_inches='tight')


def conf_mx_heat_plot(cm_dict, work_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cm_tuples = cm_dict.items()
    for ax, (model_name, conf_mx) in zip(axes, cm_tuples):
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx/row_sums
        sns.heatmap(norm_conf_mx, cmap=plt.cm.coolwarm, ax=ax, square=True,
                    vmin=0, vmax=1, annot=True, fmt='.4f', color='w', annot_kws={'fontsize': 12},
                    xticklabels=['Not readmitted', 'Readmitted'],
                    yticklabels=['Not readmitted', 'Readmitted'],
                    cbar_kws={'fraction': 0.0465, 'pad': 0.02})
        plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor", rotation=0, fontsize=9)
        plt.setp(ax.get_yticklabels(), ha="center", rotation_mode="anchor", rotation=90, fontsize=9)
        plt.tick_params(which='both', bottom=False, left=False)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(f'Heatmap for {model_name.lower()}', fontsize=11)
        ax.set_xlabel('Predicted values', fontsize=10)
        ax.set_ylabel('Actual values', fontsize=10)
    fig.suptitle('Confusion matrix heatmaps', fontsize=12)
    fig.tight_layout()
    plt.savefig(work_dir / 'plots/confusion_matrix_plots.png', dpi=288, bbox_inches='tight')
