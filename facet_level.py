from utils import facet_clustering

def facet(args, all_entities, all_vectors, fine, coarse, fine_to_coarse):

    ### Facet analysis

    for entity, vectors in all_vectors.items():
        vectors = [{v.split('_')[0] : float(v.split('_')[1])} for vec in vectors for v in vec]
        dimensions = [k for v in vectors for k in v.keys()]
        final_vectors = [vec[dim] if dim in vec.keys() else 0. for vec in vectors for dim in dimensions]
        clustered_vectors = facet_clustering(final_vectors)
        

    import pdb; pdb.set_trace()
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
