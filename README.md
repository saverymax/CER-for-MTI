# CER-for-MTI

This repository contains the code necessary to recapitulate the results of the AMIA paper, Chemical Entity Recognition for MEDLINE Indexing.   

Included are the annotations of all tools run in the paper, code for fine-tuning and running BERT and XLNet models, and scripts for performing the evaluation. Also included is the manually annotated collection, in BRAT format, and text files of all of the annotated articles. These can be used to run a CER tool, generating automatic annotations of this collection for further comparison to the manually annotated collection. 

## Training
### BERT
The code for NER in  the repository https://github.com/kyzhouhzau/BERT-NER/tree/master/old_version was used to train and run the BERT models. Modifications to the run_classifier.py script were made to adapt it the CER task. The   

### XLNet
The repository at https://github.com/stevezheng23/xlnet_extension_tf was used to train and run the XLNet model. The run_ner.py and run_ner.sh were altered to perform inference on the ChEMFAM corpus. These scripts have been replaced with the version used in the paper. The repository above  

###LSTM-CRF
The chars_lstm_lstm_crf model at https://github.com/guillaumegenthial/tf_ner was trained on CHEMDNER chemical entity mentions. It was modified to ouput BRAT.

## Running CER systems
MTI and MetaMap Lite: At this time there is no simple way to recapitulate the results of MetaMapLite or MTI. While these tools have opensource implementations, the results for this paper were generated using in-houes modifications.  

ChemDataExtractor: ChemDataExtractor can be installed and imported into Python. 
```
pip install ChemDataExtractor
```
The run_ChemDataExtractor.py script will run the system on the text, generating annotations for each article.    


## Evaluation
The run_tool_evaluation.py file can be used to run the evaluation. This will use the generated annotations from all tools to calculate all metrics. Including the -b option will run the bootstrap to compute standard errors. Including the -l option will evaluate the annotations using the Levenshtein metric, for inexact matching.

