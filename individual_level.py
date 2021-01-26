import random
import numpy
import collections
import itertools
import logging

from utils import test_clustering, write_to_file

from tqdm import tqdm

def individual(args, all_entities, all_vectors, fine, coarse, fine_to_coarse):

    ### One condition: all sentences

    ### Splitting and balancing the entity vectors
    test_data = collections.defaultdict(lambda : collections.defaultdict(list))

    for fine_cat, ent_list in fine.items():
        for ent in ent_list:
            test_data[fine_cat][ent] = all_vectors[ent]

    final_results = dict()
    for fine_cat_type, fine_dict in test_data.items():
        logging.info('Category: {}'.format(fine_cat_type))
    
        ents = [k for k in fine_dict.keys()]
        ### Pairwise category
        logging.info('Pairwise comparisons...')
        individual_combs = [c for c in itertools.combinations(ents, 2)]

        for c in tqdm(individual_combs):

            pairwise_results = collections.defaultdict(list)

            pairwise_test_data = collections.defaultdict(lambda : collections.defaultdict(list))
            ent_one = fine_dict[c[0]]
            ent_two = fine_dict[c[1]]

            ### Balancing the testdata
            max_amount = min(len(ent_one), len(ent_two))
            
            pairwise_test_data['all sentences'][c[0]] = random.sample(ent_one, k=len(ent_one))[:max_amount]
            pairwise_test_data['all sentences'][c[1]] = random.sample(ent_two, k=len(ent_two))[:max_amount]

            ### Collecting the results
            comb_results = test_clustering(args, pairwise_test_data, relevant_indices=dict(), number_of_categories=2, comparisons='{}_vs_{}'.format(c[0], c[1]))
            for result_type, results in comb_results['all sentences'].items():
                pairwise_results[result_type].append(results)

        ### Averaging the results
        averaged_pairwise = {k : 'mean: {}\tstd: {}'.format(numpy.nanmean(v), numpy.nanstd(v)) for k, v in pairwise_results.items()}
        final_results[fine_cat_type] = averaged_pairwise

    ### Finally, writing it to file
    write_to_file(args, 'individual_pairwise', final_results)
