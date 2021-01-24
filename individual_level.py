def individual(args, all_entities, all_vectors, fine, coarse, fine_to_coarse):

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
