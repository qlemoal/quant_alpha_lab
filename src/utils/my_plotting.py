# from __future__ import annotations
import matplotlib.pyplot as plt



def new_fig(fs = (12, 6)):
    f, ax = plt.subplots(figsize=fs)
    ax.grid(alpha=0.5)
    return f, ax

def finish_fig(ax, title = "", xl="", yl="", legend=True):
    ax.set_title(title)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    plt.tight_layout()
    plt.show()