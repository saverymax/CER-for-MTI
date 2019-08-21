mkdir ./checkpoints
cd checkpoints

# Download XLNet
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip -O xlnet_cased_L-12_H-768_A-12.zip
# BERT
wget https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip
# BioBERT
wget https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz
# SciBERT
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_cased.tar.gz

unzip xlnet_cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip
tar -xzvf biobert_v1.1_pubmed.tar.gz
tar -xzvf scibert_scivocab_cased.tar.gz
cd ../
