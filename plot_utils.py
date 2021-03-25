import matplotlib
import os
import numpy

from matplotlib import pyplot

def confusion_matrix(matrix, labels, very_coarse_type, score_type, plot_path):

    fig, ax = pyplot.subplots(constrained_layout=True)

    if 'facets' in plot_path:
        mode = 'facets'
        cmap = 'PuBu'
    elif 'unmasked' in plot_path:
        mode = 'unmasked'
        cmap = 'RdPu'
    else:
        mode = 'full sentence'
        cmap = 'BuGn'

    if 'all_sentences' in plot_path:
        choice = 'all sentences'
    elif 'average' in plot_path:
        choice = 'average of the sentences'
    else:
        choice = 'first sentence'

    if 'cluster' in plot_path:
        scores = list()
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                scores.append(matrix[i][j])
        mean = round(numpy.nanmean(scores), 3)
        std = round(numpy.nanstd(scores), 3)
        full_choice = 'mean {} - std {} - {}'.format(mean, std, choice)
    else:
        full_choice = choice

    mat = ax.imshow(matrix, cmap=cmap, extent=(0,len(labels),len(labels),0), vmin=0., vmax=1.)
    ax.set_title('{} \n {} -  {} - {} confusion matrix'.format(full_choice, mode, very_coarse_type, score_type), pad=10)

    ax.set_aspect(aspect='auto')

    ax.set_xticks([i+.5 for i in range(len(labels))])
    ax.set_yticks([i+.5 for i in range(len(labels))])

    ax.set_yticklabels(labels, fontsize='xx-small')
    ax.set_xticklabels(labels, fontsize='xx-small')
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                value = round(matrix[i][j], 2)
                if value < 0.4:
                    color = 'gray'
                else:
                    color = 'white'
            else:
                value = ''
                color = 'white'
            text = ax.text(j+.5, i+.5, value, ha="center", va="center", color=color)
    ax.hlines(y=[i for i in range(len(labels))], xmin=0, xmax=len(labels), color='white', linewidths=3.)
    ax.vlines(x=[i for i in range(len(labels))], ymin=0, ymax=len(labels), color='white', linewidths=3.)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    pyplot.colorbar(mat, ax=ax)
    pyplot.savefig(os.path.join(plot_path, '{}_{}_{}_{}_matrix.png'.format(very_coarse_type, score_type, choice.replace(' ', '_'), mode.replace(' ', '_'))), dpi=300)
    pyplot.clf()
    pyplot.close()
