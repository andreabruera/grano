import os
import re
import collections
import torch
import numpy
import argparse
import nltk

from transformers import BertModel, BertTokenizer
from utils.extract_word_lists import Entities
from tqdm import tqdm

def vector_to_txt(word, vector, output_file):
    output_file.write('{}\t'.format(word))
    for dimension_index, dimension_value in enumerate(vector):
        if dimension_index != len(vector)-1:
            output_file.write('{}\t'.format(dimension_value))
        else: 
            output_file.write('{}\n'.format(dimension_value))

# Create multiple vectors for Bert clustering analysis

parser = argparse.ArgumentParser()
parser.add_argument('--entities', choices=['full_wiki', 'wakeman_henson', 'eeg_stanford', 'mitchell'], default='full_wiki', help='Indicates which entities should be extracted')
args = parser.parse_args()

if args.entities == 'full_wiki':
    coarser, finer = Entities('full_wiki').words
    ents = [k for k in coarser.keys()]
    cats = [n for n in {v : 0 for k, v in finer.items()}.keys()]
else:
    ents_and_cats = Entities(args.entities).words
    ents = [k for k in ents_and_cats.keys()]
    cats = [k for k in {cat : 0 for w, cat in ents_and_cats.items()}]

words = {'ents' : ents, 'cats' : cats}

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased')

#for extraction_method in ['unmasked', 'full_sentence', 'masked']:
for extraction_method in ['full_sentence', 'masked']:

    ### Creating the folder for the word vectors
    out_folder = 'data/bert_january_2020/bert_{}_prova'.format(extraction_method)
    os.makedirs(out_folder, exist_ok=True)

    for word_type, current_words in words.items():

        for current_word in tqdm(current_words):

            ### Preparing the file and path names

            file_current_word = re.sub(' ', '_', current_word)
            txt_file = '{}.txt'.format(file_current_word)

            if word_type == 'ents' and args.entities == 'wakeman_henson':
                short_folder = re.sub('\.', '', file_current_word)[:3].lower()
                short_folder = re.sub('[^a-zA-z0-9]', '_', short_folder)[:3]
                file_name = os.path.join('/import/cogsci/andrea/dataset/corpora/wexea_annotated_wiki/ready_corpus/final_articles', short_folder, txt_file)
            else:
                short_folder = current_word[:2]
                file_name = os.path.join('/import/cogsci/andrea/dataset/corpora/wikipedia_article_by_article', short_folder, txt_file)

            ### Extracting the list of sentences for the current word
            try:
                with open(file_name) as bert_txt:
                    #bert_lines = []
                    lines = [s for l in bert_txt.readlines() for s in nltk.tokenize.sent_tokenize(l)]
                    if word_type == 'ents' and args.entities == 'wakeman_henson':
                        mention = '[[{}|'.format(current_word)
                        selected_lines = [l.strip() for l in lines if mention in l]
                        if extraction_method == 'masked':
                            selected_lines = [l.replace(mention, '[MASK][[entity|') for l in selected_lines]
                        else:
                            selected_lines = [l.replace(mention, '[[entity|') for l in selected_lines]
                        bert_lines = []
                        for line in selected_lines:
                            new_line = []
                            l_two = line.replace(']]', '[[')
                            l_three = [w for w in l_two.split('[[') if w != 'ANNOTATION']
                            l_four = [re.sub('\|\w+$', '', w) for w in l_three] 
                            l_five = [re.sub('^.+\|', '', w) for w in l_four]
                            if extraction_method == 'masked':
                                l_five = [w for w in l_five if w != current_word]
                            bert_lines.append(' '.join(l_five))
                    else:
                        common_noun = re.sub('_', ' ', current_word)
                        if '(' in common_noun:
                            common_noun = common_noun.split('(')[0].strip()
                        bert_lines = [l.strip() for l in lines if '{}'.format(common_noun) in l or '{}'.format(common_noun.lower()) in l]
                        if extraction_method == 'masked':
                            bert_lines = [l.replace(' {} '.format(common_noun.lower()), ' [MASK] ') for l in bert_lines]
                            bert_lines = [l.replace('{} '.format(common_noun), ' [MASK] ') for l in bert_lines]
                            bert_lines = [l.replace(' {}.'.format(common_noun.lower()), ' [MASK] ') for l in bert_lines]
                            bert_lines = [l.replace(' {},'.format(common_noun.lower()), ' [MASK] ') for l in bert_lines]
                        else:
                            bert_lines = [l.replace(' {} '.format(common_noun.lower()), ' [SEP] {} [SEP] '.format(common_noun.lower())) for l in bert_lines]
                            bert_lines = [l.replace('{} '.format(common_noun), ' [SEP] {} [SEP] '.format(common_noun)) for l in bert_lines]
                            bert_lines = [l.replace(' {}.'.format(common_noun.lower()), ' [SEP] {} [SEP] '.format(common_noun.lower())) for l in bert_lines]
                            bert_lines = [l.replace(' {},'.format(common_noun.lower()), ' [SEP] {} [SEP] '.format(common_noun.lower())) for l in bert_lines]


            except FileNotFoundError:
                print('impossible to extract the word vector for {}'.format(current_word))
                continue

            ### Extracting the BERT vectors

            
            bert_lines = [l for l in bert_lines if '[SEP]' in l or '[MASK]' in l]
            bert_vectors = []        
            if len(bert_lines) > 20:
                bert_lines = bert_lines[:-5] 
            for ready_line in bert_lines:
                ready_line = ready_line.replace('\t', ' ')

                input_ids = bert_tokenizer(ready_line, return_tensors='pt')
                readable_input_ids = input_ids['input_ids'][0].tolist()

                if len(readable_input_ids) <= 512:

                    if extraction_method != 'masked':

                        sep_indices = list()
                        for index, bert_id in enumerate(readable_input_ids):
                            if bert_id == 102 and index != len(readable_input_ids)-1:
                                sep_indices.append(index)
                        assert len(sep_indices)%2 == 0

                        relevant_indices = list()
                        relevant_ids = list()

                        for sep_start in range(0, len(sep_indices), 2):
                            new_window = [k-sep_start for k in range(sep_indices[sep_start], sep_indices[sep_start+1]-1)] 
                            relevant_indices.append(new_window)

                            sep_window = [k for k in range(sep_indices[sep_start]+1, sep_indices[sep_start+1])] 
                            relevant_ids.append([readable_input_ids[k] for k in sep_window])

                        input_ids = bert_tokenizer(re.sub('\[SEP\]', '', ready_line), return_tensors='pt')
                        readable_input_ids = input_ids['input_ids'][0].tolist()
                        for k_index, k in enumerate(relevant_indices):
                            try:
                                assert [readable_input_ids[i] for i in k] == relevant_ids[k_index]
                                assert 102 not in [readable_input_ids[i] for i in k]
                                assert 102 not in relevant_ids[k_index]
                            except AssertionError:
                                import pdb; pdb.set_trace()
                    else:
                        relevant_indices = [[i] for i, input_id in enumerate(readable_input_ids) if readable_input_id == 103]
                   
                    outputs = bert_model(**input_ids, return_dict=True, output_hidden_states=True, output_attentions=False)

                    assert len(readable_input_ids) == len(outputs['hidden_states'][1][0])
                    assert len(relevant_indices) >= 1
                    word_layers = list()

                    if extraction_method == 'full_sentence':
                        relevant_indices = [[i for i in range(1, len(readable_input_ids))]]
                    ### Using the first 4 layers in BERT
                    for layer in range(1, 5):
                        layer_container = list()
                        for relevant_index_list in relevant_indices:
                            for individual_index in relevant_index_list:
                                layer_container.append(outputs['hidden_states'][layer][0][individual_index].detach().numpy())
                        layer_container = numpy.average(layer_container, axis=0)
                        assert len(layer_container) == 768
                        word_layers.append(layer_container)
                    sentence_vector = numpy.average(word_layers, axis=0)
                    assert len(sentence_vector) == 768
                    bert_vectors.append(sentence_vector)
                else:
                    print(ready_line)

            with open(os.path.join(out_folder, '{}.vec'.format(file_current_word)), 'w') as o:
                for vector in bert_vectors:
                    vector_to_txt(ready_line, vector, o)
                
