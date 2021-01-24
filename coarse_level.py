def coarse(args, all_entities, all_vectors, fine, coarse, fine_to_coarse):

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
        smaller_cluster_data = [(entities_vectors[ent][0], finer[ent]) for ent in ents]
        average_cluster_data = [(numpy.average(entities_vectors[ent], axis=0), finer[ent]) for ent in ents] # average, not first sentence

        balanced_full = balance_data(all_cluster_data)
        balanced_smaller = balance_data(smaller_cluster_data)
        balanced_average = balance_data(average_cluster_data)
        #test_data = [(all_cluster_data, 'All sentences unbalanced'), (smaller_cluster_data, 'First sentence unbalanced'), (balanced_full, 'All sentences balanced'), (balanced_smaller, 'First sentence balanced')]
        #test_data = [(balanced_full, 'All sentences balanced'), (balanced_smaller, 'First sentence balanced')]
        test_data = [(balanced_full, 'All sentences balanced'), (balanced_smaller, 'First sentence balanced'), (balanced_average, 'All vectors averaged')]

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
            smaller_cluster_data = [(entities_vectors[ent][0], finer[ent]) for ent in c_data]
            average_cluster_data = [(numpy.average(entities_vectors[ent], axis=0), finer[ent]) for ent in c_data] # average, not first sentence

            balanced_full = balance_data(all_cluster_data)
            balanced_smaller = balance_data(smaller_cluster_data)
            balanced_average = balance_data(average_cluster_data)
            test_data = [(all_cluster_data, 'All sentences unbalanced'), (smaller_cluster_data, 'First sentence unbalanced'), (balanced_full, 'All sentences balanced'), (balanced_smaller, 'First sentence balanced'), (balanced_average, 'All vectors averaged')]

            current_results = test_clustering(test_data, 2)
            for i in current_results:
                coarse_results[i[0]].append(i[1])

        coarse_results = {k : [numpy.nanmean(scores[i]) for i in range(4)] for k, scores in coarse_results.items()}

        for score_tuple in coarse_results.items():
            
            category_results['Within {} pairwise'.format(coarse)].append(score_tuple)
