import numpy
import os
from tqdm import tqdm
from utils import facet_clustering

def facet(args, all_entities, all_vectors, fine, coarse, fine_to_coarse):

    ### Facet analysis

    for fine_cat, fine_names in tqdm(fine.items()):
        os.makedirs('temp/prova_facets', exist_ok=True)
        with open('temp/prova_facets/{}_roles.txt'.format(fine_cat), 'w') as o:
            #for entity, vectors in all_vectors.items():
            for entity in fine_names:
                vectors = all_vectors[entity]
                number_of_clusters = 7
                if len(vectors) > number_of_clusters*2: ### Just making sure there are enough vectors for the clustering to actually happen
                    vectors = [{v.split('_')[0] : float(v.split('_')[1]) for v in vec} for vec in vectors]
                    dimensions = list(set([k for v in vectors for k in v.keys()]))
                    final_vectors = [[vec[dim] if dim in vec.keys() else 0. for dim in dimensions] for vec in vectors]
                    clustered_vectors = facet_clustering(final_vectors, number_of_clusters=number_of_clusters, mode='kmeans')
                    averaged_vectors = {k : numpy.nanmean(v, axis=0) for k, v in clustered_vectors.items() if k!=-1}

                    five_percent = int(len(dimensions)/20)
                    interpretable_vectors = {k : [dimensions[dim[0]] for dim in sorted([(i, v) for i, v in enumerate(list(vec))], key=lambda item: item[1], reverse=True)[:five_percent]] for k, vec in averaged_vectors.items()}
                    o.write('{}\n\n'.format(entity))
                    for k, v in interpretable_vectors.items():
                        o.write('\tRole {}: {}\n\n'.format(k, v))
                    o.write('\n')

    '''
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
