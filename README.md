# Metaphor mapping completion with PTLMs 
## "MEAN: Metaphoric Erroneous ANalogies dataset for PTLMs metaphor knowledge probing"
This repository contains the MEAN dataset, scripts and notebooks needeed to reproduce the results in the submitted paper: *MEAN: Metaphoric Erroneous ANalogies dataset for PTLMs metaphor knowledge probing*. It contains:
- MEAN dataset: A dataset with selected metaphor erroneous analogies based on MetaNet data, to uncover what aspects of metaphors do LMs learn. 
- script and notebook needed to reproduce our experiment for metaphor mapping completion. 
- script and notebook to compute the baselines with static embedding models for our dataset and task.

### The dataset
The MEAN dataset consists on an expansion of a subset of MetaNet dataset. MetaNet contains 680 conceptual metaphors. From them we just selected the ones which had assigned one or more metaphor mappings between the different frame entities and which had the pattern 'A are B'. Each of the retrieved mappings is a row in our dataset. The columns contain the source and target domains of the metaphor and the source and target entities retrieved from MetaNet. The three remaining columns contained the erroneous added target entities, among which the first option has the same domain as the target domain of the metaphor but different attribute from the source entity; the second option is erroneous by having a different domain from the target domain of the metaphor but the correct attribute, and the third is has the wrong attribute and domain. At the moment our dataset contains 166 analogies made for 71 different metaphors and 100 source and target metaphor domains. In the future we aim to expand it with the rest of the metaphors in MetaNet and to other languages. 

### Metaphor mapping completion task
To run the metaphor mapping completion task `context_LM_metaphor_detection_evaluate.py` python script can be easily launched in Google Collab using `context_LM_metaphor_detection_launcher.ipynb`. It is designed to run a multiple choice task on PTLMs where given an starting sentence as `'war' is to 'cancer'` and four ending sentence such as `[ [what 'fighter' is to 'patient' ], [what 'fighter' is to 'hospital' ], [what 'fighter' is to 'teacher' ], [what 'fighter' is to 'reading' ] ]`  the model needs to select the ending sentence (with target and source lexical entry correspondences) that best fits the beginning (with metaphor's source and target domain verbalization). 


### Baselines
To obtain the baselines `static_LM_metaphor_detection_evaluate.py` python script can be easily launched in Google Collab using `static_LM_metaphor_detection_launcher.ipynb` . 
