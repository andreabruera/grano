import os
import re
import collections
import numpy

def prepare_file_name(entity):

    entity_one = re.sub(' ', '_', entity)
    entity_file_name = '{}.vec'.format(entity_one)
    return entity_file_name

def extract_word_vectors(entity):
    #path = '/import/cogsci/andrea/dataset/bert_january_2020/bert_unmasked_prova'
    path = '/import/cogsci/andrea/dataset/word_vectors/bert_january_2020/bert_full_sentence_prova'
    entity_file_name = prepare_file_name(entity)

    try:
        with open(os.path.join(path, entity_file_name)) as entity_file:
            entity_lines = [l for l in entity_file.readlines()]
        stop = False
    except FileNotFoundError:
        mapping_dict = {'Object.vec' : 'Physical_object.vec'}
        try:
            with open(os.path.join(path, mapping_dict[entity_file_name])) as entity_file:
                entity_lines = [l for l in entity_file.readlines()]
            stop = False
        except KeyError:
            stop = True
        
    if not stop:

        lines = [l.split('\t')[0] for l in entity_lines]
        entity_vectors = [numpy.array(l.strip().split('\t')[1:], dtype=numpy.single) for l in entity_lines]
        if len(entity_vectors) > 24:
            entity_vectors = entity_vectors[:24]
    else:
        entity_vectors = list()

    return entity_vectors

class Entities:

    def __init__(self, required_words):

        self.base_folder = '/import/cogsci/andrea/github/names_count'

        if required_words == 'wakeman_henson':
            self.words = self.wakeman_henson()
        elif required_words == 'full_wiki':
            self.words = self.full_wiki()
        elif required_words == 'men':
            self.words = self.men()
        elif required_words == 'eeg_stanford':
            self.words = self.eeg_stanford()
        elif required_words == 'mitchell':
            self.words = self.mitchell()
        elif required_words == 'uk':
            self.words = self.uk()
        elif required_words == 'stopwords':
            self.words = self.stopwords()

    def vectors(self, words):
        word_vectors = {w : extract_word_vectors(w) for w in words if re.sub('[0-9]', '', str(w)) != ''}
        return word_vectors
    '''
    def category_vectors(self):
        category_vectors = {cat : extract_word_vectors(cat) for w, cat in (self.words).items() if cat != ''}
        return category_vectors
    '''
       
    def wakeman_henson(self):
        with open('resources/wakeman_henson_stimuli.txt') as input_file:
            lines = [l.strip().split('\t') for l in input_file.readlines()]
        names = [l[1] for l in lines if len(l) > 2]
        names_and_cats = {l[1] : l[2] for l in lines if len(l) > 2}
        return names_and_cats

    def eeg_stanford(self):
        #with open('/import/cogsci/andrea/github/fame/data/resources/eeg_data_ids.txt') as ids_txt:
        with open('/import/cogsci/andrea/github/fame/resources/eeg_stanford_stimuli_ids.txt') as ids_txt:
            raw_lines = [l.strip().split('\t') for l in ids_txt.readlines()]
        lines = [l for l in raw_lines]
        words_and_cats = {l[1] : l[2] for l in lines}
        #mapping_dictionary = {'Object' : 'Physical object', 'Japan' : 'Physical object'}
        #words_and_cats = {k : mapping_dictionary[v] if v in mapping_dictionary.keys() else v for k, v in words_and_cats.items()}

        return words_and_cats

    def mitchell(self):
        with open('/import/cogsci/andrea/github/fame/resources/mitchell_words_and_cats.txt') as mitchell_file:
            raw_lines = [l.strip().split('\t') for l in mitchell_file.readlines()]
            words_and_cats = {l[0].capitalize() : l[1].capitalize() for l in raw_lines}
            mapping_dictionary = {'Manmade' : 'Physical object', 'Buildpart' : '', 'Bodypart' : 'Human body'}
            words_and_cats = {k : mapping_dictionary[v] if v in mapping_dictionary.keys() else v for k, v in words_and_cats.items()}
        return words_and_cats

    def full_wiki(self):
        coarser = collections.defaultdict(str)
        finer = collections.defaultdict(str)

        for root, direct, files in os.walk(os.path.join(self.base_folder, 'wikipedia_entities_list')):
            for f in files:
                with open(os.path.join(self.base_folder, root, f), 'r') as entities_file:
                    all_lines = [l.strip().split('\t') for l in entities_file.readlines() if '\tPlace' in l or '\tPerson' in l]
                for l in all_lines:
                    #person_words.append(l[0]) if l[1] == 'Person' else places_words.append(l[0])
                    name = re.sub('_', ' ', l[0])
                    if len(l) > 3:
                        try:
                            coarser[name] = l[2]
                            mapping = {'Fictional' : 'Character (arts)', 'Neighborhood' : 'Neighbourhood','Sports' : 'Athlete', 'Lake' : 'Body of water', 'Sea' : 'Body of water', 'River' : 'Body of water'}
                            #finer_cat = l[3] if l[3] != 'Sea' and l[3] != 'River' and l[3] != 'Lake' else 'Body of water'
                            finer[name] = l[3] if l[3] not in mapping.keys() else mapping[l[3]]
                        except KeyError:
                            print('Couldn\'t find {}'.format(name))

        return [coarser, finer]

    ### From here on, probably useless functions

    def full_wiki_for_clustering(self):
        
        out_dict = collections.defaultdict(list)

        with open(os.path.join(self.base_folder, 'models/TransE/entity_map.txt')) as mapping_file:
            id_map = collections.defaultdict(str)
            for l in mapping_file.readlines():
                line = l.strip().split('\t')
                id_map[line[0]] = line[1]

        for root, direct, files in os.walk(os.path.join(self.base_folder, 'wikipedia_entities_list')):
            for f in files:
                with open(os.path.join(self.base_folder, root, f), 'r') as entities_file:
                    all_lines = [l.strip().split('\t') for l in entities_file.readlines() if '\tPlace' in l or '\tPerson' in l]
                for l in all_lines:
                    #person_words.append(l[0]) if l[1] == 'Person' else places_words.append(l[0])
                    name = re.sub('_', ' ', l[0])
                    if len(l) > 3:
                        try:
                            unified_id = id_map[name]
                            finer_cat = l[3] if l[3] != 'Sea' and l[3] != 'River' and l[3] != 'Lake' else 'Body of water'
                            out_dict[name] = (l[2], finer_cat, unified_id, int(l[1]))
                        except KeyError:
                            print('Couldn\'t find {}'.format(name))

        # Collecting categories

        categories = collections.defaultdict(lambda: collections.defaultdict(int))
        for ent, desc in out_dict.items():
            coarse = desc[0]
            fine = desc[1]
            categories[coarse][fine] += 1
               
        return out_dict, categories

    def wakeman_henson_old(self):

        with open(os.path.join(self.base_folder, 'resources/wiki_stimuli.txt')) as ids_file:
            wiki2vec_ids = [((re.sub('.bmp', '', l)).strip('\n')).split('\t') for l in ids_file.readlines()]
            category_info = {l[0] : l[2:] for l in wiki2vec_ids if len(l) > 2}
            fmri2wiki = {l[0] : l[1] for l in wiki2vec_ids if len(l) > 2}
        with open(os.path.join(self.base_folder, 'resources/transe_stimuli.txt')) as ids_file:
            transe_ids = [((re.sub('.bmp', '', l)).strip('\n')).split('\t') for l in ids_file.readlines()]
            fmri2transe = {l[0] : l[1] for l in transe_ids if len(l) > 2}
        assert len(wiki2vec_ids) == len(transe_ids)
        assert [k for k in fmri2wiki.keys()] == [k for k in fmri2transe.keys()]

        return fmri2wiki, fmri2transe, category_info

    def men(self):
        words = []
        with open(os.path.join(self.base_folder, 'resources/men.txt'), 'r') as men_file:
            all_lines = [l.split()[:2] for l in men_file.readlines()]
        all_words = [i for i in {re.sub('_.', '', w) : '' for l in all_lines for w in l}.keys()]
        return all_words

    def uk(self):
        files = ['people', 'places']
        entity_words = []
        generic_words = []
        wikidata_identifiers = []
        for f in files:
            with open(os.path.join(self.base_folder, 'resources/uk_{}.txt'.format(f)), 'r') as uk_file:
                all_lines = [l.strip().split('\t') for l in uk_file.readlines()]
            for l in all_lines:
                entity_words.append(l[0])
                generic_words.append(l[1])
                wikidata_identifiers.append(l[2])
        return entity_words, generic_words, wikidata_identifiers


    def stopwords(self):
        words = []
        with open(os.path.join(self.base_folder, 'resources/stopwords.txt'), 'r') as stopwords_file:
            all_words = [l.strip('\n' ) for l in stopwords_file.readlines()[1:] if len(l) >= 5]
        return all_words
