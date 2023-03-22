
'''
Script for metaphor evaluation with LM

Use case:

python3 context_LM_metaphor_detection_evaluate.py \
    --model 'roberta-base'   \
	--metaphor_file '/path/to/mean_dataset'   \
    --dir_output_results '/path/to/output_dir/' \
	--start_template "' <W1> ' is to ' <W2> '" \
	--end_template  " what ' <W3> ' is to ' <W4> '." \
	--nepochs 5 \
	--batch_size 8 \
    --kfolds 10

The script produces two result files:
 - 'res_<time_ini_exec>.txt'. It contain three lines: The arguments of the script, 
   the start and end time of the execution, and the obtained accuracy in each fold
   and the mean accuracy.
 - 'res_<time_ini_exec>.csv'. A csv file where each line contains the information
   of a metaphor: source and target domains, source element, the four target element
   alternatives, the type of the alternatives, the real and predited target element,
   and the logits produce by the model for the four alternatives.

  
'''

import os
from datetime import datetime
import numpy as np
import random
import re

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
import evaluate

import logging
import argparse

from sklearn.model_selection import KFold

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
	Copied from https://huggingface.co/docs/transformers/tasks/multiple_choice
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def transform_dataset(one_row, num_choices):
    dict_row_names={0:'Gold', 1:'SameFrameDifferentAttribute',2:'DiffFrameSameAtt', 3:'DiffFrameDiffAtt'}
    row = {k:v for k, (_,v) in enumerate(one_row.items())}
    new_indxs = random.sample([i for i in range(num_choices)], num_choices)
    new_order = [row[3+i] for i in new_indxs]
    new_type_alt = [dict_row_names[i] for i in new_indxs]
    correct = new_indxs.index(0)
    res_dict = {'MetSource':row[0], 'MetTarget':row[1], 'SourceRole':row[2]}
    for i in range(num_choices):
        res_dict['alt_'+str(i)] = new_order[i]
    for i in range(num_choices):
        res_dict['type_alt_'+str(i)] = new_type_alt[i]
    res_dict['labels'] = correct
    return res_dict		
	
def verbalize_metaphor(four_words, template, tokenizer):
    verb = re.sub('<W1>', four_words[0].lower(), template)
    verb = re.sub('<W2>', four_words[1].lower(), verb)
    verb = re.sub('<W3>', four_words[2].lower(), verb)
    verb = re.sub('<W4>', four_words[3].lower(), verb)
    verb = re.sub('<SEP>', tokenizer.sep_token, verb)
    verb = re.sub('<MASK>', tokenizer.mask_token, verb)
    verb = re.sub('_', ' ', verb)
    return verb

def verbalize_metaphor_row(row, template, tokenizer, col_name):
    lista = [(row['MetSource'],row['MetTarget'], row['SourceRole'], row['alt_'+str(i)]) for i in range(4)]
    verbs_met = [verbalize_metaphor(four_words, template, tokenizer) for four_words in lista]
    return {col_name:verbs_met}

def preprocess_function(examples, tokenizer):
    first_sentences = sum(examples['verb_start'], [])
    if 'verb_end' in examples:
        second_sentences = sum(examples['verb_end'], [])
    if 'verb_end' in examples:
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True,)
    else:
        tokenized_examples = tokenizer(first_sentences, truncation=True,)
    inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    return inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metrics.compute(predictions=predictions, references=labels)
	

	
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='Train and test models for metaphor detection.')
parser.add_argument("-model", "--model", required=True, help="Path to a LM model")
parser.add_argument("-metfile", "--metaphor_file", required=True, help="Path to the metaphor file")
parser.add_argument("-o", "--dir_output_results", required=True, help="Directory to save the results")
parser.add_argument("-stt", "--start_template", required=True, help="Beggining template for the verbalization of a metaphor")
parser.add_argument("-ett", "--end_template", required=False, help="Ending template for the verbalization of a metaphor")
parser.add_argument("-e", "--nepochs", required=True, type=int, help="Number training epochs")
parser.add_argument("-b", "--batch_size", required=True, type=int, help="Batch size")
parser.add_argument("-kf", "--kfolds", default=10, type=int, help="Batch size")

# parameters
args = parser.parse_args()
model_name = args.model
metaphor_file = args.metaphor_file
output_dir = args.dir_output_results
template_start = args.start_template
template_end = []
if args.end_template is not None:
    template_end = args.end_template
batch_size = args.batch_size
total_epochs = args.nepochs
kfolds = args.kfolds

#create output dir, if it does not exist
try:
    os.makedirs(output_dir)
except:
   pass
   
# dictionary containing the start and end execution times
now = datetime.now()
dates = {'ini':now.strftime("%Y-%m-%d_%H-%M-%S")}

# download metric: accuracy
metrics = evaluate.load("accuracy") 

# download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# read metaphor data
metaphor_data = load_dataset("csv", data_files = metaphor_file, sep=';', skiprows=1)

# add random alternatives and their types
metaphor_data_mod = metaphor_data.map(transform_dataset, fn_kwargs={'num_choices':4})

# add to the dataset the verbalized templates
metaphor_data_mod = metaphor_data_mod.map(verbalize_metaphor_row, fn_kwargs={'template':template_start, 'tokenizer':tokenizer, 'col_name':'verb_start'} )
if len(template_end) > 0:
	metaphor_data_mod = metaphor_data_mod.map(verbalize_metaphor_row, fn_kwargs={'template':template_end, 'tokenizer':tokenizer, 'col_name':'verb_end'} )
	
# shuffle data
metaphor_data_mod['train'] = metaphor_data_mod['train'].shuffle()

# encoded data: inputs for the model
encoded_metaphor_data_mod = metaphor_data_mod.map(preprocess_function, batched=True, fn_kwargs={'tokenizer':tokenizer})

# training arguments
training_args = TrainingArguments(
    output_dir="my_checkpoint",
    evaluation_strategy="no",
    #save_strategy="epoch",
    #load_best_model_at_end=True,
    learning_rate=2e-5,
    optim="adamw_torch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=total_epochs,
    weight_decay=0.01,
)

# list to store the results for each fold
res_datasets = []
res_metrics = {}

# for a K-fold cross-validation
folds = KFold(n_splits=kfolds)

for i, (train_index, test_index) in enumerate(folds.split(encoded_metaphor_data_mod['train'])):
    logging.info("Training: Cross validation, step: " + str(i+1) + "/" + str(kfolds))
    data_train = Dataset.from_dict(encoded_metaphor_data_mod['train'][train_index])
    data_test = Dataset.from_dict(encoded_metaphor_data_mod['train'][test_index])
	
    # load the model
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
	
    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        #eval_dataset=data_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # predict test metaphors
    preds = trainer.predict(data_test)
    pred_labels = np.argmax(preds.predictions, axis=1)
    fold_metrics = metrics.compute(predictions=pred_labels, references=preds.label_ids)
    print(fold_metrics)
    
    res_metrics['fold'+str(i).zfill(2)] = fold_metrics
    	
    col_names = data_test.features.keys()
    remove_col_names = [col for col in col_names if re.match("TargetRole|Diff|Same|verb|attention_mask|input_ids", col)]
    res_fold_dataset = data_test.remove_columns(remove_col_names)
    res_fold_dataset = res_fold_dataset.add_column('pred', pred_labels)
    for i, col_logits in enumerate(preds.predictions.T):
        res_fold_dataset = res_fold_dataset.add_column('logit' + str(i), col_logits)
		
    res_datasets.append(res_fold_dataset) 	
	
#end execution time
now = datetime.now()
dates['end'] = now.strftime("%Y-%m-%d_%H-%M-%S")

#write results files
res_file_name = output_dir + '/res_' + dates['ini']

res_dataset_final = concatenate_datasets(res_datasets)
res_dataset_final.to_csv(res_file_name + '.csv')

preds_total = res_dataset_final['pred']
real_total = res_dataset_final['labels']
res_metrics['all'] = metrics.compute(predictions=preds_total, references=real_total)

with open(res_file_name + '.txt', 'w') as f:
    print(vars(args), file=f)
    print(dates, file=f)
    print(res_metrics, file=f) 


