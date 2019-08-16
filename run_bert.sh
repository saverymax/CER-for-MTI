#!/bin/bash

# Run chemdner BERT 
python BERT_annotator.py \
    --task_name="NER" \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --do_CDI=False \
    --do_lower_case=False \
    --citation_dir=./data/ChEMFAM_corpus/*.txt \
    --vocab_file=./checkpoints/cased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=./checkpoints/cased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=./model_output/bert_chemdner_output/model.ckpt-4349 \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --learning_rate=3e-5 \
    --num_train_epochs=3.0 \
    --output_dir=./model_output/bert_chemdner_output/ \
    --output_preds=results/bert_chemdner_test.txt \
    --n_labels=2

# Run BERT trained with BioC Gene mentions
python BERT_annotator.py \
    --task_name="NER" \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --do_CDI=False \
    --use_crf=False \
    --do_lower_case=False \
    --citation_dir=./data/ChEMFAM_corpus/*.txt \
    --vocab_file=./checkpoints/cased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=./checkpoints/cased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=./model_output/bert_gm_output/model.ckpt-5757 \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --learning_rate=3e-5 \
    --num_train_epochs=3.0 \
    --output_dir=./model_output/bert_gm_output/ \
    --output_preds=results/bert_gene_mention_test.txt \
    --n_labels=2 

# Run biobert gene mentions
python BERT_annotator.py \
    --task_name="NER" \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --do_lower_case=False \
    --citation_dir=./data/ChEMFAM_corpus/*.txt \
    --vocab_file=./checkpoints/biobert_v1.1_pubmed/vocab.txt \
    --bert_config_file=./checkpoints/biobert_v1.1_pubmed/bert_config.json \
    --init_checkpoint=./model_output/biobert_gm_output/model.ckpt-5757 \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --learning_rate=3e-5 \
    --num_train_epochs=3.0 \
    --output_dir=./model_output/biobert_gm_output/ \
    --output_preds=results/biobert_gene_mentions_test.txt \
    --n_labels=2 

#Run biobert chemdner
python BERT_annotator.py \
    --task_name="NER" \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --do_lower_case=False \
    --citation_dir=./data/ChEMFAM_corpus/*.txt \
    --vocab_file=./checkpoints/biobert_v1.1_pubmed/vocab.txt \
    --bert_config_file=./checkpoints/biobert_v1.1_pubmed/bert_config.json \
    --init_checkpoint=./model_output/biobert_chemdner_output/model.ckpt-4349 \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --learning_rate=3e-5 \
    --num_train_epochs=3.0 \
    --output_dir=./model_output/biobert_chemdner_output/ \
    --output_preds=results/biobert_chemdner_test.txt \
    --n_labels=2 

# Run chemdner SciBERT
python BERT_annotator.py \
    --task_name="NER" \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --do_lower_case=False \
    --citation_dir=./data/ChEMFAM_corpus/*.txt \
    --vocab_file=./checkpoints/scibert_scivocab_cased/vocab.txt \
    --bert_config_file=./checkpoints/scibert_scivocab_cased/bert_config.json \
    --init_checkpoint=./model_output/scibert_chemdner_output/model.ckpt-4349 \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --learning_rate=3e-5 \
    --num_train_epochs=3.0 \
    --output_dir=./model_output/scibert_chemdner_output/ \
    --output_preds=results/scibert_chemdner_test.txt \
    --n_labels=2 

# Run gene mention SciBERT
python BERT_annotator.py \
    --task_name="NER" \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --do_lower_case=False \
    --citation_dir=./data/ChEMFAM_corpus/*.txt \
    --vocab_file=./checkpoints/scibert_scivocab_cased/vocab.txt \
    --bert_config_file=./checkpoints/scibert_scivocab_cased/bert_config.json \
    --init_checkpoint=./model_output/scibert_gm_output/model.ckpt-5757 \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --learning_rate=3e-5 \
    --num_train_epochs=3.0 \
    --output_dir=./model_output/scibert_gm_output/ \
    --output_preds=results/scibert_gene_mention_test.txt \
    --n_labels=2 

cp ./model_output/scibert_gm_output/results/scibert_gene_mention_test.txt ./data/tool_annotations
cp ./model_output/scibert_chemdner_output/results/scibert_chemdner_test.txt ./data/tool_annotations
cp ./model_output/biobert_chemdner_output/results/biobert_chemdner_test.txt ./data/tool_annotations
cp ./model_output/biobert_gm_output/results/biobert_gene_mentions_test.txt ./data/tool_annotations
cp ./model_output/bert_gm_output/results/bert_gene_mention_test.txt ./data/tool_annotations
cp ./model_output/bert_chemdner_output/results/bert_chemdner_test.txt ./data/tool_annotations
