import matplotlib
import os

from matplotlib import pyplot

def confusion_matrix(matrix, labels, very_coarse_type, score_type, plot_path):

    fig, ax = pyplot.subplots(constrained_layout=True)

    mat = ax.imshow(matrix, cmap='PuBu', extent=(0,5,5,0), vmin=0., vmax=1.)
    ax.set_title('{} confusion matrix'.format(score_type), pad=10)

    ax.set_aspect(aspect='auto')

    ax.set_xticks([i+.5 for i in range(len(labels))])
    ax.set_yticks([i+.5 for i in range(len(labels))])

    ax.set_yticklabels(labels, fontsize='xx-small')
    ax.set_xticklabels(labels, fontsize='xx-small')
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j+.5, i+.5, round(matrix[i][j], 2), ha="center", va="center", color="w")
    ax.hlines(y=[i for i in range(len(labels))], xmin=0, xmax=len(labels), color='white', linewidths=3.)
    ax.vlines(x=[i for i in range(len(labels))], ymin=0, ymax=len(labels), color='white', linewidths=3.)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    pyplot.colorbar(mat, ax=ax)
    pyplot.savefig(os.path.join(plot_path, '{}_{}_matrix.png'.format(very_coarse_type, score_type)), dpi=300)
    pyplot.clf()
    pyplot.close()
