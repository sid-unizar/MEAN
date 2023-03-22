"""
Script for metaphor detection with static word embeddings

Use case:

python3 static_LM_metaphor_detection_evaluate.py \
	--model 'word2vec-google-news-300'   \
	--metaphor_file '/path/to/mean_dataset'   \
    --dir_output_results '/path/to/output_dir/'

Another examples of models that can be used:
  - 'word2vec-google-news-300'
  - 'glove-wiki-gigaword-300'
  - 'fasttext-wiki-news-subwords-300'

The script produces two result files:
 - 'res_<time_ini_exec>.txt'. It contain three lines: The arguments of the script, 
   the start and end time of the execution, and the obtained accuracy. 
   and the mean accuracy.
 - 'res_<time_ini_exec>.csv'. A csv file where each line contains the information
   of a metaphor: source and target domains, source element, the four target element
   alternatives, the type of the alternatives, the real and predited target element.
"""

import os
import numpy as np
import gensim.downloader
from gensim.models.keyedvectors import KeyedVectors
from datasets import load_dataset

import argparse
from datetime import datetime
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Metaphor detectection with static word embeddings')
parser.add_argument("-model", "--model", required=True, help="Name of a static language model")
parser.add_argument("-metfile", "--metaphor_file", required=True, help="Path to the metaphor file")
parser.add_argument("-o", "--dir_output_results", required=True, help="Directory to save the results")

# parameters
args = parser.parse_args()
model_name = args.model
metaphor_file = args.metaphor_file
output_dir = args.dir_output_results

#create output dir, if it does not exist
try:
    os.makedirs(output_dir)
except:
   pass

# dictionary containing the start and end execution times
now = datetime.now()
dates = {'ini':now.strftime("%Y-%m-%d_%H-%M-%S")}

# download/load the parameters model to a KeyedVectors
logging.info('Downloading/loading the model')
static_model = gensim.downloader.load(model_name)

# load the metaphor dataset to a DatasetDict object
metaphor_data = load_dataset("csv", data_files = metaphor_file, sep=';', skiprows=1)

# to avoid OOV, add new words and their regarding calculate embedding to the model
# following the statregy: 
#  - for a word, remove the last character upto the word is in the model
#  - for a multiword: calculate the mean of the word embeddings in the multiword

logging.info("Adding new words")
keys_vectors_add_to_model = {}

for row in metaphor_data['train']:
    print(row)
    for i, (col_name,word) in enumerate(row.items()):
        if not static_model.has_index_for(word.lower()) and not static_model.has_index_for(word.lower().capitalize()) and not word.lower() in keys_vectors_add_to_model:
            multiwords_aux = word.lower().split("_")
            list_keys_in_model = []
            for w in multiwords_aux:
                for i in range(len(w)):
                    if static_model.has_index_for(w[0:(len(w)-i)].lower()):
                        list_keys_in_model.append(w[0:(len(w)-i)].lower())
                        break
                    elif static_model.has_index_for(w[0:(len(w)-i)].lower().capitalize()):
                        list_keys_in_model.append(w[0:(len(w)-i)].lower().capitalize())
                        break
            # checkpoint            
            if len(list_keys_in_model) == 0:
                raise Exception("no model keys in list_keys_in_model")
            # checkpoint
            elif len(list_keys_in_model) != len(multiwords_aux) :
                raise Exception("list_keys_in_model len != multiwords_aux len")
            elif len(list_keys_in_model) == 1:
                keys_vectors_add_to_model[word.lower()] = static_model.get_vector(list_keys_in_model[0])
            else:
                new_vector = static_model.get_mean_vector(list_keys_in_model)
                keys_vectors_add_to_model[word.lower()] = static_model.get_mean_vector(list_keys_in_model)
                
# add new words in the model
list_new_keys = []
list_new_vectors = []
for k, v in keys_vectors_add_to_model.items():
    list_new_keys.append(k)
    list_new_vectors.append(v)

if len(list_new_keys) > 0 :
    static_model.add_vectors(list_new_keys, list_new_vectors)
    static_model.fill_norms(True)

logging.info("Finished: Adding new words")

# fake words are added to the model to process
# the metaphors later more easily. It is added
# one word for each metaphor in the dataset.
# The fake word is "met<i>" (where <i> is the row 
# number) and the regarding embedding is: 
#   X = MetTarget_Embedding - MetSource_embedding + SourceRole_Embedding
#    MetSource  --- MetTarget
#    SourceRole --- X
logging.info("Calculating fake words (one fake word for each metaphor in the dataset)")
key_vector_metaphors = {}
for number_met, row in enumerate(metaphor_data['train']):
    new_vector = None
    for i, (col_name,word) in enumerate(row.items()):
        try:
            v = static_model.get_vector(word.lower(), norm=True)
        except:
            v = static_model.get_vector(word.lower().capitalize(), norm=True)
        if i == 0:
            new_vector = -v
        elif i == 1 or i == 2:
            new_vector += v
    if new_vector is None:
        print(number_met)
    key_vector_metaphors['met'+str(number_met)] = new_vector

list_met_ids = []
list_new_vectors_met = []
for k, v in key_vector_metaphors.items():
    list_met_ids.append(k)
    list_new_vectors_met.append(v)

if len(list_met_ids) > 0 :
    static_model.add_vectors(list_met_ids, list_new_vectors_met)
    static_model.fill_norms(True)
    
# inference: calculate the accuracy
logging.info('Calculating results')
pred = []
real = []
total = 0
for number_met, row in enumerate(metaphor_data['train']):
    key = 'met' + str(i)
    lista_alternativas = []
    for i, (col_name, word) in enumerate(row.items()):
        if i > 2:
            if static_model.has_index_for(word.lower()):
                lista_alternativas.append(word.lower())
            elif static_model.has_index_for(word.lower().capitalize()):
                lista_alternativas.append(word.lower().capitalize())
            else:
                # checkpoint            
                raise Exception("Something is wrong. Word: " + word + " is OOV")
    best_alt = static_model.most_similar_to_given(key, lista_alternativas)
    pred.append(best_alt)
    real.append(lista_alternativas[0])
    if best_alt == lista_alternativas[0]:
        total = total + 1

accuracy = total/metaphor_data['train'].num_rows
res_metrics = {'accuracy':accuracy}

#end execution time
now = datetime.now()
dates['end'] = now.strftime("%Y-%m-%d_%H-%M-%S")

#write results files
res_file_name = output_dir + '/res_' + dates['ini']
with open(res_file_name + '.txt', 'w') as f:
    print(vars(args), file=f)
    print(dates, file=f)
    print(res_metrics, file=f) 

metaphor_data['train'] = metaphor_data['train'].add_column('pred', pred)
metaphor_data['train'] = metaphor_data['train'].add_column('real', real)

metaphor_data['train'].to_csv(res_file_name + '.csv')
    
