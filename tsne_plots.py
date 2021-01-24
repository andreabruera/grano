from matplotlib import pyplot

from sklearn.manifold import TSNE

import collections

import numpy

import matplotlib.cm as cm

def compute_tsne(vectors):

    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    tsne_embeddings = tsne_model_en_2d.fit_transform(vectors)

    return tsne_embeddings

def tsne_plot_words(title, dict_one, dict_two, labels, filename=None):

    words_one = [k for k in dict_one.keys()]
    words_two = [k for k in dict_two.keys()]

    colors = get_colors_dict(words_one, words_two)

    vecs_one = [v for k, v in dict_one.items()]
    vecs_two = [v for k, v in dict_two.items()]

    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)

    embeddings = tsne_model_en_2d.fit_transform(vecs_one + vecs_two)
    words = words_one + words_two
    final_labels = [labels[0] for word in words_one] + [labels[1] for word in words_two]
    
    pyplot.figure(figsize=(16, 9))

    c = 1
    for embedding, word, label in zip(embeddings, words, final_labels):
        if c == 1:
        #pyplot.scatter(embedding[0], embedding[1], c=(colors[word].reshape(1, colors[word].shape[0])), alpha=1, edgecolors='k', s=120, label=label)
            pyplot.scatter(embedding[0], embedding[1], c=(colors[word].reshape(1, colors[word].shape[0])), alpha=1, edgecolors='k', s=120, label=labels[0])
        elif c == len(embeddings):
            pyplot.scatter(embedding[0], embedding[1], c=(colors[word].reshape(1, colors[word].shape[0])), alpha=1, edgecolors='k', s=120, label=labels[1])
        else:
            pyplot.scatter(embedding[0], embedding[1], c=(colors[word].reshape(1, colors[word].shape[0])), alpha=1, edgecolors='k', s=120)
        #pyplot.annotate(word, alpha=1, xy=(embedding[0], embedding[1]), xytext=(10, 7), textcoords='offset points', ha='center', va='bottom', size=12)
        c += 1

    pyplot.title(title, fontsize='xx-large', fontweight='bold', pad = 15.0)
    pyplot.legend()

    if filename:
        pyplot.savefig(filename, format='png', dpi=300, bbox_inches='tight')

def get_colors_dict(category_one, category_two):

    color_dict = collections.defaultdict(numpy.ndarray)

    collection = {'category_one' : category_one, 'category_two' : category_two}

    for category, content in collection.items():
        if category == 'category_one': 
            colors_gen = cm.Wistia(numpy.linspace(0, 1, (len(category_two))))
        if category == 'category_two':
            colors_gen = cm.winter(numpy.linspace(1, 0, (len(category_two))))
        c = 0
        for name in content:
            if name not in color_dict.keys():
                color_dict[name] = colors_gen[c]
                c += 1

    return color_dict
