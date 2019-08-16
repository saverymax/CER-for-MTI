#!/bin/bash

training_data=./data/training_data/
mkdir model_output
mkdir model_output/bert_chemdner_output
mkdir model_output/bert_gm_output
mkdir model_output/biobert_chemdner_output
mkdir model_output/biobert_gm_output
mkdir model_output/scibert_chemdner_output
mkdir model_output/scibert_gm_output
mkdir model_output/results
mkdir model_output/bert_chemdner_output/results
mkdir model_output/bert_gm_output/results
mkdir model_output/biobert_chemdner_output/results
mkdir model_output/biobert_gm_output/results
mkdir model_output/scibert_chemdner_output/results
mkdir model_output/scibert_gm_output/results

# Bert 512 cased, chemdner 
python BERT_annotator.py --task_name="NER" --do_train=True --do_eval=False --do_predict=False --do_lower_case=False --data_dir=${training_data} --vocab_file=./checkpoints/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=./checkpoints/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=./checkpoints/cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=512 --train_batch_size=4 --learning_rate=3e-5 --num_train_epochs=3.0 --output_dir=./model_output/bert_chemdner_output/ --n_labels=2 --training_file=chemdner_CDI_training.txt

# Bert 512 casedm BioC gene mentions
python BERT_annotator.py --task_name="NER" --do_train=True --do_eval=False --do_predict=False --do_lower_case=False --data_dir=${training_data} --vocab_file=./checkpoints/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=./checkpoints/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=./checkpoints/cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=512 --train_batch_size=4 --learning_rate=3e-5 --num_train_epochs=3.0 --output_dir=./model_output/bert_gm_output/ --n_labels=2 --training_file=BioC_GM_training.txt


# SciBERT cased, on chemdner
python BERT_annotator.py --task_name="NER" --do_train=True --do_eval=False --do_predict=False --do_lower_case=False --data_dir=${training_data} --vocab_file=./checkpoints/scibert_scivocab_cased/vocab.txt --bert_config_file=./checkpoints/scibert_scivocab_cased/bert_config.json --init_checkpoint=./checkpoints/scibert_scivocab_cased/bert_model.ckpt --max_seq_length=512 --train_batch_size=4 --learning_rate=3e-5 --num_train_epochs=3.0 --output_dir=./model_output/scibert_chemdner_output/ --n_labels=2 --training_file=chemdner_CDI_training.txt

# SciBERT cased, on gene mentions
python BERT_annotator.py --task_name="NER" --do_train=True --do_eval=False --do_predict=False --do_lower_case=False --data_dir=${training_data} --vocab_file=./checkpoints/scibert_scivocab_cased/vocab.txt --bert_config_file=./checkpoints/scibert_scivocab_cased/bert_config.json --init_checkpoint=./checkpoints/scibert_scivocab_cased/bert_model.ckpt --max_seq_length=512 --train_batch_size=4 --learning_rate=3e-5 --num_train_epochs=3.0 --output_dir=./model_output/scibert_gm_output/ --n_labels=2 --training_file=BioC_GM_training.txt

# BioBERT cased, on chemdner
python BERT_annotator.py --task_name="NER" --do_train=True --do_eval=False --do_predict=False --do_lower_case=False --data_dir=${training_data} --vocab_file=./checkpoints/biobert_v1.1_pubmed/vocab.txt --bert_config_file=./checkpoints/biobert_v1.1_pubmed/bert_config.json --init_checkpoint=./checkpoints/biobert_v1.1_pubmed/model.ckpt-1000000 --max_seq_length=512 --train_batch_size=4 --learning_rate=3e-5 --num_train_epochs=3.0 --output_dir=./model_output/biobert_chemdner_output/ --n_labels=2 --training_file=chemdner_CDI_training.txt

# BioBERT on gene mentions
python BERT_annotator.py --task_name="NER" --do_train=True --do_eval=False --do_predict=False --do_lower_case=False --data_dir=${training_data} --vocab_file=./checkpoints/biobert_v1.1_pubmed/vocab.txt --bert_config_file=./checkpoints/biobert_v1.1_pubmed/bert_config.json --init_checkpoint=./checkpoints/biobert_v1.1_pubmed/model.ckpt-1000000 --max_seq_length=512 --train_batch_size=4 --learning_rate=3e-5 --num_train_epochs=3.0 --output_dir=./model_output/biobert_gm_output/ --n_labels=2 --training_file=BioC_GM_training.txt
