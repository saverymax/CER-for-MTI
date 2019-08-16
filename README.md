# CER-for-MTI

This repository contains the code necessary to recapitulate the results of the AMIA summit paper, Chemical Entity Recognition for MEDLINE Indexing.   

Included are the annotations of all tools run in the paper, code for fine-tuning and running BERT and XLNet models, and scripts for performing the evaluation. Also included is the manually annotated collection, in BRAT format, and text files of all of the annotated articles.

## Chemical Entity Mentions for Assessment of MTI (ChEMFAM) corpus
Included is the ChEMFAM corpus, located in the data/ChEMFAM_corpus directory. The .ann files (BRAT format) and the .txt files of the articles have been shared. In the .txt files, the first line contains the title of the article and the second line contains the abstract.   

The guidelines for annotations are available as a .docx file, ChEMFAM_Annotation_Guidelines.docx.

## Dependencies
Before training and running evaluation, it is recommended to create a virtual python environment, with python 3.6.8. 
For example 
```
conda create --name chemfam_env python=3.6.8
```
The dependencies can be installed with 
```
pip install -r requirements.txt
```
This will install the following packages
tf_metrics   
sentencepiece   
skmetrics   
leven   
tensorflow version 1.12.2   
numpy version 1.16.1   

## Training
The annotations from trained systems are already provided in this repository, in the data/tool_annotations directory. If you just want to see the results, go to the evaluation section of the README. To generate these yourself, the models must be trained on entity mentions and run on the ChEMFAM corpus. Instructions to train models are included here.

Training was performed with a GeForce GTX 1080 Ti, with 11GB of RAM.  

### BERT
The code for NER in the repository https://github.com/kyzhouhzau/BERT-NER/tree/master/old_version was used as reference to write the BERT_annotator.py script.

To train just the bert models, run the train_bert.sh script. This will generate BERT, SciBERT, and BioBERT models, trained on the BC4CHEMD and BC2GM data (one model trained on one dataset, six models total).

### XLNet
The code for NER in the repository https://github.com/stevezheng23/xlnet_extension_tf was used as reference to write the XLNet_annotator.py script.

To train the XLNet models, run the train_xlnet.sh script.

## Running CER systems
 To run all CER systems on the ChEMFAM corpus, run the run_models.sh script. Instructions for individual models are below.

### ChemDataExtractor 
ChemDataExtractor can be installed and imported into Python. 
```
pip install ChemDataExtractor
```
The run_ChemDataExtractor.py script will run the system on the text, generating annotations for each article.    

### PubTator Central
Pubtator can be accessed at https://www.ncbi.nlm.nih.gov/research/pubtator/index.html. Upload the pmids_to_annotate.txt file to the collection manager, and download the results, placing them in the tool_annotations directory. 

NEED TO ADD BIT TO NOT HARDCODE THIS

### BERT
All BERT models, including SciBERT and BioBERT, can be run with the run_bert.sh script. This will generate predictions for chemicals in the ChEMFAM corpus.

### XLNet
XLNet models can be run with the run_xlnet.sh script.

### MTI and MetaMapLite
At this time there is no simple way to recapitulate the results of MetaMapLite or MTI. While these tools have opensource implementations, the results for this paper were generated using in-houes modifications.  

### ChemListem and LSTM-CRF.
No code is provided to run these models. However, the code can be found at https://bitbucket.org/rscapplications/chemlistem/src/master/ and https://github.com/guillaumegenthial/tf_ner. Additionally, there are many open source implementations of these types of LSTM/CNN/CRF models.

## Evaluation
After train_models.sh and run_models.sh have been run, or the individual models above have been trained and run, the run_tool_evaluation.py file can be used to run the evaluation. This will use the annotations from all tools to calculate F1-score, recall, and precision. Including the -b option will run bootstrap to compute standard errors. Including the -l option will evaluate the annotations using the Levenshtein metric, for inexact matching.   

The results for each model can be viewed in the results_printouts directory. The results will be saved to one of four files, depending on the CLI options used:
results_tool_evaluation.txt for results calculated using exact matching   
results_tool_evaluation_bootstrap.txt for results calculated using exact matching and bootstrap to generate the standard error    
results_tool_evaluation_leven.txt for results calculated using relaxed matching criteria (levenshtein distance normalized by string length)   
results_tool_evaluation_leven_bootstrap.txt for standard errors of relazed matching results   

Additionally annotation sets for each tool can be found in the data/annotation_sets directory. If the -l option has been used, Levenshtein measurements for each tool for each entity can be found in result_printouts/levenshtein_measurements.txt

## Repository references
https://github.com/google-research/bert
https://github.com/zihangdai/xlnet
https://github.com/stevezheng23/xlnet_extension_tf
https://github.com/kyzhouhzau/BERT-NER
