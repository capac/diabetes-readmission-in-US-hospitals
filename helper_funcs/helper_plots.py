#! /usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt

# plot settings
row_length_in_px = 12
column_length_in_px = 4
nrows = 1
ncols = 3

plt.style.use("style_files/lineplot-style.mplstyle")


def roc_curve_plot_with_auc(rates_dict, work_dir):
    fig, axes = plt.subplots(nrows, ncols, figsize=(row_length_in_px,
                                                    column_length_in_px))
    rates_tuples = rates_dict.items()
    for ax, (model_name, rates_list) in zip(axes.flatten(), rates_tuples):
        fpr, tpr, model_roc_auc = rates_list
        ax.plot(fpr, tpr, 'b.-', label='AUC: {:.4f}'.format(model_roc_auc))
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlim([-0.02, 1.0])
        ax.set_ylim([0.0, 1.02])
        plt.setp(ax.get_xticklabels(), fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        if len(model_name) > 3:
            ax.set_title(f'ROC curve for {model_name.lower()}', fontsize=10)
        else:
            ax.set_title(f'ROC curve for {model_name.upper()}', fontsize=10)
        ax.legend(loc='lower right', fontsize=12)
    fig.suptitle('Receiver operating characteristic curves',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    plt.savefig(work_dir / 'plots/auc_plots.png',
                bbox_inches='tight')


def conf_mx_heat_plot(cm_dict, work_dir):
    fig, axes = plt.subplots(nrows, ncols, figsize=(row_length_in_px,
                                                    column_length_in_px))
    cm_tuples = cm_dict.items()
    for ax, (model_name, conf_mx) in zip(axes.flatten(), cm_tuples):
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx/row_sums
        sns.heatmap(norm_conf_mx, cmap=plt.cm.coolwarm, ax=ax, square=True,
                    vmin=0, vmax=1, annot=True, fmt='.4f', color='w',
                    annot_kws={'fontsize': 12},
                    xticklabels=['Not readmitted', 'Readmitted'],
                    yticklabels=['Not readmitted', 'Readmitted'],
                    cbar_kws={'fraction': 0.0465, 'pad': 0.02})
        plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor",
                 rotation=0, fontsize=9)
        plt.setp(ax.get_yticklabels(), ha="center", rotation_mode="anchor",
                 rotation=90, fontsize=9)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        plt.tick_params(which='both', bottom=False, left=False)
        if len(model_name) > 3:
            ax.set_title(f'Heatmap for {model_name.lower()}', fontsize=10)
        else:
            ax.set_title(f'Heatmap for {model_name.upper()}', fontsize=10)
        ax.set_xlabel('Predicted values', fontsize=10)
        ax.set_ylabel('Actual values', fontsize=10)
    fig.suptitle('Confusion matrix heatmaps', fontsize=11, fontweight='bold')
    # fig.tight_layout()
    plt.savefig(work_dir / 'plots/confusion_matrix_plots.png',
                bbox_inches='tight')
