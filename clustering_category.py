import os
import numpy
import collections
import re
import random
import itertools
import time
import argparse
import pickle
import numpy

from sklearn import metrics
from sklearn.cluster import KMeans
from utils.extract_word_lists import Entities
from utils.evaluation import cos
from tqdm import tqdm
from scipy.stats import pearsonr

def prepare_file_name(entity):

    entity_one = re.sub(' ', '_', entity)
    entity_file_name = '{}.vec'.format(entity_one)
    return entity_file_name

def extract_word_vectors(args, entity):
    path = '/import/cogsci/andrea/github/fame/data/bert_january_2020/bert_{}_prova'.format(vector_extraction_mode)
    entity_file_name = prepare_file_name(entity)
    with open(os.path.join(path, entity_file_name)) as entity_file:
        entity_vectors = [numpy.array(l.strip().split('\t')[1:], dtype=numpy.single) for l in entity_file.readlines()]
    return entity_vectors

def balance_data(data):
    categories = collections.defaultdict(list)
    for sample in data:
        categories[sample[1]].append(sample[0])
    shuffled_categories = {label : random.sample(v, k=len(v)) for label, v in categories.items()}
    max_per_category = min([len(v) for k, v in shuffled_categories.items()])
    balanced_data = {label : v[:max_per_category] for label, v in shuffled_categories.items()}
    out_data = [(vec, label) for label, vecs in balanced_data.items() for vec in vecs]
    return out_data

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
    
def test_clustering(data, categories):

    results = []

    for training_samples in data:

        samples = training_samples[0]
        data_type = training_samples[1]
        kmeans = KMeans(n_clusters=categories, random_state=0)
        kmeans.fit([sample[0] for sample in samples])
        predicted_labels = kmeans.labels_
        purity_score, majority_per_class = purity(predicted_labels, [k[1] for k in samples])
        v_score = (metrics.v_measure_score([k[1] for k in samples], kmeans.labels_))
        homogeneity_score = (metrics.homogeneity_score([k[1] for k in samples], kmeans.labels_))
        completeness_score = (metrics.completeness_score([k[1] for k in samples], kmeans.labels_))
        
        current_results = [data_type, [v_score, homogeneity_score, completeness_score, purity_score]]
        results.append(current_results)
        if len(samples) == 150:
            import pdb; pdb.set_trace()
        #print(current_results)

    return results

parser = argparse.ArgumentParser()
parser.add_argument('--masked', action='store_true', default=False, help='Uses vectors obtained from masked words instead of normal mentions')
args = parser.parse_args()

time_now = time.strftime('%d_%b_%H_%M', time.gmtime())
vector_extraction_mode = 'masked' if args.masked else 'unmasked'

Ents = Entities('full_wiki')
'''
with open('temp/full_wiki_stats.txt', 'w') as o:
    for coarse, finer_counter in categories.items():
        o.write('{}\n\n'.format(coarse))
        for fine, count in finer_counter.items():
            o.write('{}: {}'.format(fine, count))
        o.write('\n\n\n') 
entities_vectors = collections.defaultdict(list)

for ent in tqdm(entities_list.keys()):
    try: 
        entities_vectors[ent] = extract_word_vectors(args, ent)
    except FileNotFoundError:
        print(ent)
'''
coarser, finer = Ents.words


### Cleaning up the list of entities from those which have no actual vectors
exclude_list = ['Country', 'City', 'Area', 'Politics', 'Body of water', 'Literature', 'Music', 'Monument', 'Film', 'Athlete']

entities_vectors = Ents.vectors([k for k in coarser.keys() if finer[k] in exclude_list])

entities_vectors = {k : v for k, v in entities_vectors.items() if len(v) > 0}
entities = [k for k in entities_vectors.keys()]

coarser = {k : coarser[k] for k in entities}
finer = {k : finer[k] for k in entities}
fine_categories = [k for k  in {v : 0 for k, v in finer.items()}.keys()]
fine_to_coarse = {fine : coarser[n] for n, fine in finer.items()}
fine_vectors = Ents.vectors(fine_categories)
fine_vectors = {k : v for k, v in fine_vectors.items() if len(v) > 0}

coarse_full = coarser.copy()
coarse_full.update(fine_to_coarse)
both = entities_vectors.copy()
both.update(fine_vectors)

### Coarse people/place clustering
category_results = collections.defaultdict(list)
conditions = [('individuals', entities_vectors), ('common', fine_vectors), ('both', both)]

for condition in conditions:

    condition_name = condition[0]
    condition_data = condition[1]

    print('Now evaluating clustering for the coarse categories - {} words'.format(condition_name))

    all_cluster_data = [(v, coarse_full[ent]) for ent, vecs in condition_data.items() for v in vecs]
    #smaller_cluster_data = [(vecs[0], coarse_full[ent]) for ent, vecs in condition_data.items()]
    smaller_cluster_data = [(numpy.average(vecs, axis=0), coarse_full[ent]) for ent, vecs in condition_data.items()] # average, not first sentence

    balanced_full = balance_data(all_cluster_data)
    balanced_smaller = balance_data(smaller_cluster_data)

    #test_data = [(all_cluster_data, 'All sentences unbalanced'), (smaller_cluster_data, 'First sentence unbalanced'), (balanced_full, 'All sentences balanced'), (balanced_smaller, 'First sentence balanced')]
    test_data = [(balanced_full, 'All sentences balanced'), (balanced_smaller, 'First sentence balanced')]

    category_results['Coarse_{}'.format(condition_name)] = test_clustering(test_data, 2)

### Finer grained people/places clustering

# Setting up the coarser categories

categories = collections.defaultdict(list)
categories['people'] = [k for k, v in coarser.items() if v == 'Person']
categories['place'] = [k for k, v in coarser.items() if v == 'Place']

# All together

print('Now evaluating clustering for the finer categories, taken all together')
for coarse, ents in tqdm(categories.items()):
    number_categories = len({v : 0 for k, v in finer.items() if k in ents})
    print('Category: {}\t- Number of categories: {}'.format(coarse, number_categories))
    all_cluster_data = [(v, finer[ent]) for ent in ents for v in entities_vectors[ent]]
    #smaller_cluster_data = [(entities_vectors[ent][0], finer[ent]) for ent in ents]
    smaller_cluster_data = [(numpy.average(entities_vectors[ent], axis=0), finer[ent]) for ent in ents] # average, not first sentence

    balanced_full = balance_data(all_cluster_data)
    balanced_smaller = balance_data(smaller_cluster_data)
    #test_data = [(all_cluster_data, 'All sentences unbalanced'), (smaller_cluster_data, 'First sentence unbalanced'), (balanced_full, 'All sentences balanced'), (balanced_smaller, 'First sentence balanced')]
    test_data = [(balanced_full, 'All sentences balanced'), (balanced_smaller, 'First sentence balanced')]

    category_results['Within {} all together'.format(coarse)] = test_clustering(test_data, number_categories)

# Pairwise

print('Now evaluating clustering for the finer categories, taken two at a time')
for coarse, ents in tqdm(categories.items()):
    fine_cats = collections.defaultdict(list)

    for e in ents:
        fine_cats[finer[e]].append(e)
    cats_combs = itertools.combinations([k for k in fine_cats.keys()], 2)

    coarse_results = collections.defaultdict(list)
    for c in cats_combs:

        c_data = fine_cats[c[0]] + fine_cats[c[1]]
        all_cluster_data = [(v, finer[ent]) for ent in c_data for v in entities_vectors[ent]]
        #smaller_cluster_data = [(entities_vectors[ent][0], finer[ent]) for ent in c_data]
        smaller_cluster_data = [(numpy.average(entities_vectors[ent], axis=0), finer[ent]) for ent in c_data] # average, not first sentence

        balanced_full = balance_data(all_cluster_data)
        balanced_smaller = balance_data(smaller_cluster_data)
        test_data = [(all_cluster_data, 'All sentences unbalanced'), (smaller_cluster_data, 'First sentence unbalanced'), (balanced_full, 'All sentences balanced'), (balanced_smaller, 'First sentence balanced')]

        current_results = test_clustering(test_data, 2)
        for i in current_results:
            coarse_results[i[0]].append(i[1])

    coarse_results = {k : [numpy.nanmean(scores[i]) for i in range(4)] for k, scores in coarse_results.items()}

    for score_tuple in coarse_results.items():
        
        category_results['Within {} pairwise'.format(coarse)].append(score_tuple)


### Writing results to file

with open('temp/category_evaluation_{}_{}.txt'.format(vector_extraction_mode, time_now), 'w') as o:
    for coarse, within_results in category_results.items():
        o.write('{}\n\n'.format(coarse))
        for result in within_results:
            o.write('{}:\n\tv score: {}\n\thomogeneity: {}\n\tcompleteness: {}\n\tpurity: {}\n'.format(result[0], result[1][0], result[1][1], result[1][2], result[1][3]))
        o.write('\n\n\n')
import pdb; pdb.set_trace()
### Writing results to file

with open('temp/category_evaluation_{}_{}.txt'.format(vector_extraction_mode, time_now), 'a') as o:
    for coarse, within_results in category_results.items():
        o.write('{}\n\n'.format(coarse))
        for result in within_results:
            o.write('{}: {}\n'.format(result[0], result[1]))
        o.write('\n\n\n')
'''
### Pairwise individual vs individual evaluation

print('Now evaluating clustering for couples of individuals within the finer categories')

results = collections.defaultdict(lambda: collections.defaultdict(list))

for coarse, fine_counter in categories.items():
    fine = [k for k, v in fine_counter.items() if v > 5]
    print('Clustering of individuals within {}'.format(coarse))
    for f in tqdm(fine):
        f_data = {ent : vecs  for ent, vecs in entities_vectors.items() if entities_list[ent][1] == f}
        combs = itertools.combinations([k for k in f_data.keys()], 2)
        for c in combs:
            #print(c)
            full_data = [(sample, 0) for sample in f_data[c[0]]] + [(sample, 1) for sample in f_data[c[1]]]
            balanced_data = balance_data(full_data)
            current_results = test_clustering([[full_data, 'Full data'], [balanced_data, 'Balanced data']], 2)
            results[coarse][f].append(current_results)

### Pickling results, so as to be able to analyse results

to_be_pickled = {k : v for k, v in results.items()}
del results
with open('temp/pickled_individual_evaluation_{}_{}.pkl'.format(vector_extraction_mode, time_now), 'wb') as o:
    pickle.dump(to_be_pickled, o) 

### Writing results for the individual vs individual analysis:

with open('temp/pickled_individual_evaluation_masked_21_Oct_21_31.pkl', 'rb') as i:
    to_be_pickled = pickle.load(i)

#with open('temp/individual_vs_individual_{}_{}.txt'.format(vector_extraction_mode, time_now), 'w') as o:
with open('temp/individual_vs_individual_masked_25_10.txt', 'w') as o:
    for coarse, finer in to_be_pickled.items():
        overall_results = collections.defaultdict(list)
        o.write('{}\n\n\n'.format(coarse))
        for fine, scores in finer.items():
            fine_results = collections.defaultdict(list)
            o.write('{}\n'.format(fine))
            for both_scores in scores:
                for score in both_scores:
                    fine_results[score[0]].append(score[1][0])
            for data_type, filtered_scores in fine_results.items():
                current_mean = numpy.nanmean(filtered_scores)
                current_std = numpy.nanstd(filtered_scores)
                o.write('{}: mean {} - std {}\n'.format(data_type, current_mean, current_std))
                overall_results[data_type].append(current_mean)
            o.write('\n\n')
        for data_type, scores in overall_results.items():
            o.write('General mean for {}, {}: {}\n'.format(coarse, data_type, numpy.nanmean(scores)))
        o.write('\n\n')

### Facet analysis

rsa_full = collections.defaultdict(dict)

for e, vecs in entities_vectors.items():
    rsa_ent = collections.defaultdict(list)
    facet = 0
    for vec in vecs:
        current_ent = []
        facet += 1
        for vec_two in vecs:
            current_ent.append(pearsonr(vec, vec_two)[0])
            #current_ent.append(cos(vec, vec_two))
        rsa_ent[facet] = numpy.array(current_ent, numpy.single)
    combs = itertools.combinations([k for k in rsa_ent.keys()], 2)
    rsa_results = [pearsonr(rsa_ent[comb[0]], rsa_ent[comb[1]])[0] for comb in combs]
    #rsa_results = [cos(rsa_ent[comb[0]], rsa_ent[comb[1]]) for comb in combs]
    print('Entity: {}\nAverage RSA similarity: {}\tStd: {}\n\n'.format(e, numpy.nanmean(rsa_results), numpy.nanstd(rsa_results)))
    rsa_full[e] = rsa_ent

with open('temp/pickled_rsa_facet_evaluation_{}_{}.pkl'.format(vector_extraction_mode, time_now), 'wb') as o:
    pickle.dump(rsa_full, o) 
'''
