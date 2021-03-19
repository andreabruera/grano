import collections
import numpy
import random
import logging
import os

from tsne_plots import compute_tsne

from matplotlib import pyplot
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfTransformer

def fine_cat_evaluation(data_type, coarse_dicts, relevant_indices, predicted_labels, majority_per_class):

    fine_cat_indices = dict()
    c = 0
    for coarse_cat, vecs in coarse_dicts.items():
        if c == 0:
            fine_cat_indices[coarse_cat] = [i for i in range(relevant_indices[data_type], len(vecs))]
            c += 1
        else:
            fine_cat_indices[coarse_cat] = [i for i in range(relevant_indices[data_type]+len(vecs), len(vecs)*2)]

    ### Collecting the fine category clustering result
    fine_cat_clustering = list()

    for coarse_cat, indices in fine_cat_indices.items():
        predictions = [predicted_labels[i] for i in indices]
        gold_majority_class = majority_per_class[coarse_cat]
        accuracy = len([k for k in predictions if k==gold_majority_class]) / len(predictions)
        fine_cat_clustering.append(accuracy)

    fine_cat_accuracy = numpy.nanmean(fine_cat_clustering)

    return fine_cat_indices, fine_cat_accuracy

def plot_tsne(args, comparisons, data_type, samples, golden_labels, fine_cat_indices):
    path = prepare_folder(args, data_type)
    colors = tsne_colors(args)
    tsne_samples = compute_tsne(samples)
    ### Splitting the data for visualization
    tsne_data = collections.defaultdict(list)
    for label_index, label in enumerate(golden_labels):
        if label_index in fine_cat_indices[label]:
            fine_label = '{} (categories)'.format(label)
        else:
            fine_label = '{} (entities)'.format(label)
        tsne_data[fine_label].append(tsne_samples[label_index])

    fig, ax = pyplot.subplots()
    for label, vectors in tsne_data.items():
        ax.scatter([k[0] for k in vectors], [k[1] for k in vectors], label=label, color=colors[label], edgecolors='white', linewidths=.25, s=6.)
    ax.legend()
    ax.set_title('{} - {}'.format(args.granularity_level, data_type))
    pyplot.savefig(os.path.join(path, '{}_tsne_plot.png'.format(comparisons)), dpi=300)

def tsne_colors(args):
    if args.granularity_level == 'very_coarse':
        colors = {'Person (entities)' : 'goldenrod',
                  'Person (categories)' : 'gold',
                  'Place (entities)' : 'teal',
                  'Place (categories)' : 'lightseagreen'}

    elif args.granularity_level == 'coarse':
        colors = {'Actor (entities)' : 'c',
                  'Athlete (entities)' : 'm',
                  'Musician (entities)' : 'y',
                  'Writer (entities)' : 'darkgray',
                  'Politician (entities)' : 'orange',
                  'City (entities)' : 'c',
                  'Area (entities)' : 'm',
                  'Body of water (entities)' : 'y',
                  'Country (entities)' : 'lightgray',
                  'Monument (entities)' : 'orange'}
    return colors

def prepare_folder(args, data_type):
    
    path = os.path.join('cluster_results', args.vector_mode, args.granularity_level, data_type.replace(' ', '_'))
    os.makedirs(path, exist_ok=True)

    return path

def purity(predictions, real_classes):

    majority_per_class = collections.defaultdict(int)

    class_predictions = collections.defaultdict(list)
    for pred, real in zip(predictions, real_classes):
        class_predictions[real].append(pred)
    purity_scores_container = []

    for real_class, pred in class_predictions.items():
        counter = collections.defaultdict(int)
        for p in pred:
            counter[p] += 1
        majority_class = [k[0] for k in sorted(counter.items(), reverse=True, key=lambda count: count[1])][0]
        majority_per_class[real_class] = majority_class
        purity_scores_container.append(sum([1 for k in pred if k == majority_class]))

    purity_score = sum(purity_scores_container) / len(predictions)

    return purity_score, majority_per_class
    
def test_clustering(args, data, relevant_indices, number_of_categories, comparisons):

    results = collections.defaultdict(lambda : collections.defaultdict(float))

    for data_type, coarse_dicts in data.items():

        if args.granularity_level != 'individual':
            logging.info('Now clustering data in mode: {}'.format(data_type))
        ### Preparing the data and recording the indices at which the vectors for the finer categories are
        samples = list()
        golden_labels = list()

        for coarse_cat, vecs in coarse_dicts.items():
            for v in vecs:
                samples.append(v)
                golden_labels.append(coarse_cat)

        ### Clustering and evaluating
        kmeans = KMeans(n_clusters=number_of_categories, random_state=0)
        kmeans.fit(samples)
        predicted_labels = kmeans.labels_
        purity_score, majority_per_class = purity(predicted_labels, golden_labels)
        v_score = (metrics.v_measure_score(golden_labels, kmeans.labels_))
        homogeneity_score = (metrics.homogeneity_score(golden_labels, kmeans.labels_))
        completeness_score = (metrics.completeness_score(golden_labels, kmeans.labels_))

        results[data_type]['purity'] = purity_score
        results[data_type]['v-score'] = v_score
        results[data_type]['homogeneity'] = homogeneity_score
        results[data_type]['completeness'] = completeness_score

        if args.granularity_level == 'very_coarse':
            fine_cat_indices, fine_cat_accuracy = fine_cat_evaluation(data_type, coarse_dicts, relevant_indices, predicted_labels, majority_per_class)
            results[data_type]['finer category accuracy'] = fine_cat_accuracy

        else:
            fine_cat_indices = {k : [] for k in golden_labels}

        if not args.granularity_level == 'individual':

            ### Plotting the tsne visualization
            logging.info('Now plotting tsne in mode: {}'.format(data_type))
            #plot_tsne(args, comparisons, data_type, samples, golden_labels, fine_cat_indices)

            ### Writing to file
            #write_to_file(args, comparisons, results)

    return results

def facet_clustering(vectors, number_of_clusters, mode='kmeans'):

    vectors = TfidfTransformer().fit_transform(X=vectors)
    vectors = vectors.todense().tolist()
    if mode  == 'kmeans':
        clusters = KMeans(n_clusters=number_of_clusters, random_state=0).fit(vectors)
    elif mode == 'dbscan':
        clusters = DBSCAN(min_samples=3).fit(vectors)
    labels = clusters.labels_
    #labels = k.labels_
    #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #n_noise_ = list(labels).count(-1)

    #print('Estimated number of clusters: %d' % n_clusters_)
    #print('Estimated number of noise points: %d' % n_noise_)

    labeled_vectors = collections.defaultdict(list)
    for label, vector in zip(labels, vectors):
        labeled_vectors[label+1 if label!=-1 else -1].append(vector)
    
    return labeled_vectors

def write_to_file(args, comparisons, results_dict):

    ### Writing results to file

    for data_type, data_dict in results_dict.items():

        data_type_path = prepare_folder(args, data_type)

        with open(os.path.join(data_type_path, '{}_results_breakdown.txt'.format(comparisons)), 'w') as o:
            o.write('{} evaluation\t-\t{}\n\n'.format(args.granularity_level, data_type))
            for score_name, score in data_dict.items():
                o.write('{}\t{}\n'.format(score_name, score))
