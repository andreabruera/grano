import argparse
import os
import numpy
import collections
import re
import random
import itertools
import time
import pickle
import numpy
import logging
import pickle

from very_coarse_level import very_coarse
from coarse_level import coarse_level
from individual_level import individual
from facet_level import facet

from tqdm import tqdm
from scipy.stats import pearsonr

def clustering_analysis(args, all_entities, all_vectors, fine, coarse, fine_to_coarse):

    if args.granularity_level == 'very_coarse':
        results = very_coarse(args, all_entities, all_vectors, fine, coarse, fine_to_coarse)
    elif args.granularity_level == 'coarse':
        results = coarse_level(args, all_entities, all_vectors, fine, coarse, fine_to_coarse)
    elif args.granularity_level == 'individual':
        results = individual(args, all_entities, all_vectors, fine, coarse, fine_to_coarse)
    elif args.granularity_level == 'facet':
        results = facet(args, all_entities, all_vectors, fine, coarse, fine_to_coarse)

    return results

parser = argparse.ArgumentParser()
parser.add_argument('--vector_mode', required=True, choices=['masked', 'unmasked', 'full_sentence', 'facets'], help='Specifies which vectors to use')
parser.add_argument('--granularity_level', required=True, choices=['very_coarse', 'coarse', 'individual', 'facet'], help='Indicates at which level of granularity analyses should be carried out')
parser.add_argument('--entity_central_path', default='/import/cogsci/andrea/github', help='Indicates where to look for the entity_central package')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.INFO)

### Adding to the local paths the path where to find the utilities from entity_central
import sys
sys.path.append(args.entity_central_path)

from entity_central.extract_word_lists import Entities
from entity_central.read_vectors import EntityVectors

logging.info('Loading entities and vectors')
#entities = Entities('full_wiki')
#vectors = EntityVectors(entities.word_categories, 'bert', args.vector_mode)

with open('pickles/very_quick_pickle_{}.pkl'.format(args.vector_mode), 'rb') as o:
    #pickle.dump((entities, vectors), o)
    entities, vectors = pickle.load(o)

logging.info('Cleaning up entities and vectors')
### Reduce entity list
include_list = ['Country', 'City', 'Area', 'Politician', 'Body of water', 'Writer', 'Musician', 'Monument', 'Actor', 'Athlete']
to_be_deleted = vectors.to_be_deleted
all_entities = dict()
fine = collections.defaultdict(list)
coarse = collections.defaultdict(list)

for k, v in entities.words.items():
    if k not in to_be_deleted and v[0] in include_list:
        all_entities[k] = v
        fine[v[0]].append(k)
        coarse[v[1]].append(k)
all_vectors = vectors.vectors

fine_to_coarse = {v[0] : v[1] for k, v in all_entities.items()}

clustering_analysis(args, all_entities, all_vectors, fine, coarse, fine_to_coarse)
