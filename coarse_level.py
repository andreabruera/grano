import os
import random
import numpy
import collections
import itertools
import logging

from utils import test_clustering, prepare_folder
from plot_utils import confusion_matrix

def coarse_level(args, all_entities, all_vectors, fine, coarse, fine_to_coarse):

    ### Three conditions: all sentences, first per-entity sentence, average of the per-entity sentences

    ### Splitting and balancing the entity vectors
    full_ent_data = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(list)))

    for fine_cat, ent_list in fine.items():
        coarse_cat = fine_to_coarse[fine_cat]
        for ent in ent_list:
            full_ent_data[coarse_cat]['average of the sentences'][fine_cat].append(numpy.average(all_vectors[ent], axis=0))
            full_ent_data[coarse_cat]['definitional sentence'][fine_cat].append(all_vectors[ent][0])
            for ent_vec in all_vectors[ent]:
                full_ent_data[coarse_cat]['all sentences'][fine_cat].append(ent_vec)

    ### Creating the test dictionary and recording the index from which fine category vectors start
    test_data = dict()
    relevant_indices = collections.defaultdict(list)

    for very_coarse_cat, very_coarse_dict in full_ent_data.items():

        data_type_data = dict()
        data_type_indices = dict()

        for data_type, fine_dict in very_coarse_dict.items():
            max_amount = min([len(v) for k, v in fine_dict.items()])
            balanced_dict = {k : random.sample(v, k=len(v))[:max_amount] for k, v in fine_dict.items()}
            data_type_data[data_type] = balanced_dict
            data_type_indices[data_type] = max_amount

        test_data[very_coarse_cat] = data_type_data
        relevant_indices[very_coarse_cat] = data_type_indices

    ### Splitting and balancing the category vectors
    full_cat_data = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(list)))

    for fine_cat, coarse_cat in fine_to_coarse.items():
        full_cat_data[coarse_cat]['average of the sentences'][fine_cat].append(numpy.average(all_vectors[fine_cat], axis=0))
        full_cat_data[coarse_cat]['definitional sentence'][fine_cat].append(all_vectors[fine_cat][0])
        for cat_vec in all_vectors[fine_cat]:
            full_cat_data[coarse_cat]['all sentences'][fine_cat].append(cat_vec)

    ### Adding the category vectors to the test dictionary
    for very_coarse_cat, very_coarse_dict in full_cat_data.items():

        data_type_data = dict()
        data_type_indices = dict()

        for data_type, fine_dict in very_coarse_dict.items():
            max_amount = min([len(v) for k, v in fine_dict.items()])
            balanced_dict = {k : random.sample(v, k=len(v))[:max_amount] for k, v in fine_dict.items()}
            for fine_cat, vecs in balanced_dict.items():
                for vec in vecs:
                    test_data[very_coarse_cat][data_type][fine_cat].append(vec)

    all_results = dict()

    for very_coarse_type, data in test_data.items():
        logging.info('Category: {}'.format(very_coarse_type))
        logging.info('All fine categories...'.format(very_coarse_type))
        fine_categories = {k for k, v in fine_to_coarse.items() if v==very_coarse_type}
        number_of_categories = len(fine_categories)

        ### All fine categories together
        test_clustering(args, data, relevant_indices, number_of_categories, comparisons='all_within_{}'.format(very_coarse_type))
    
        ### Pairwise category
        logging.info('Pairwise comparisons...'.format(very_coarse_type))
        current_results = collections.defaultdict(lambda : collections.defaultdict(list))
        fine_cats_combs = [c for c in itertools.combinations(fine_categories, 2)]

        for c in fine_cats_combs:
            pairwise_test_data = dict()
            for data_type, fine_dict in data.items():
                data_type_dict = dict()
                data_type_dict[c[0]] = fine_dict[c[0]]
                data_type_dict[c[1]] = fine_dict[c[1]]
                pairwise_test_data[data_type] = data_type_dict
            comb_results = test_clustering(args, pairwise_test_data, relevant_indices, number_of_categories, comparisons='{}_vs_{}'.format(c[0], c[1]))

            current_results[c[0]][c[1]] = comb_results
            current_results[c[1]][c[0]] = comb_results

        ### Just adding the 0.0 for the square matrix
        for f in fine_categories:
            f_dict = collections.defaultdict(lambda : collections.defaultdict(float))
            for data_type in data.keys():
                f_dict[data_type]['purity'] = 0.0 
                f_dict[data_type]['v_score'] = 0.0 
            current_results[f][f] = f_dict

        matrix_results = collections.defaultdict(lambda : collections.defaultdict(list))
        for f_one in fine_categories:
            pair_results = collections.defaultdict(lambda : collections.defaultdict(list))
            for f_two in fine_categories:

                all_res = current_results[f_one][f_two]

                for data_type, data_res in all_res.items():

                    purity = data_res['purity']
                    v_score = data_res['v-score']

                    pair_results[data_type]['purity'].append(purity)       
                    pair_results[data_type]['v-score'].append(v_score)       

            for data_type, data_dict in pair_results.items():
                for score_type, score_list in data_dict.items():
                    assert len(score_list) == len(fine_categories)
                    matrix_results[data_type][score_type].append(score_list)
        
        for data_type, data_dict in matrix_results.items():
            plot_path = prepare_folder(args, data_type)
            os.makedirs(plot_path, exist_ok=True)
            for score_type, matrix in data_dict.items():
                confusion_matrix(matrix, fine_categories, very_coarse_type, score_type, plot_path)

