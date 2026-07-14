# from __future__ import annotations
import matplotlib.pyplot as plt


#  General figure commands

def new_fig(fs = (8, 6)):
    f, ax = plt.subplots(figsize=fs)
    return f, ax

def finish_fig(ax, title = "", xl="", yl="", legend=True):
    ax.set_title(title)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(alpha=0.5)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    plt.tight_layout()
    plt.show()


#  Correlation plot

def plot_corr_mat(corr_mat, fs=(7, 6), cmap='PRGn', vmin=-1, vmax=1):
    f = plt.figure(figsize=fs)
    plt.matshow(corr_mat, fignum=f.number, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(range(corr_mat.shape[0]), corr_mat.columns, fontsize=11, rotation=45, ha='left')
    plt.yticks(range(corr_mat.shape[0]), corr_mat.columns, fontsize=11, rotation=0)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=13)
    plt.show()