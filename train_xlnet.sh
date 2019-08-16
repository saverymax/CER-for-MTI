#!/bin/bash

data_dir=./data/training_data

mkdir ./model_output/xlnet_chemdner
mkdir ./model_output/xlnet_gm
mkdir ./model_output/xlnet_chemdner/results
mkdir ./model_output/xlnet_gm/results

# train xl net base on chemdner
python XLNet_annotator.py \
    --spiece_model_file=./checkpoints/xlnet_cased_L-12_H-768_A-12/spiece.model \
    --model_config_path=./checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
    --init_checkpoint=./checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt \
    --task_name=CER \
    --output_dir=./model_output/xlnet_chemdner \
    --model_dir=./model_output/xlnet_chemdner \
    --data_dir=${data_dir} \
    --input_file=chemdner_CDI_training.txt \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --learning_rate=3e-5 \
    --adam_epsilon=1e-6 \
    --train_steps=8000 \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --do_train=True \
    --do_eval=False \
    --do_predict=False \
    --do_export=False \
    --lower_case=False \

# Train xl net base on gene mentions
python XLNet_annotator.py \
    --spiece_model_file=./checkpoints/xlnet_cased_L-12_H-768_A-12/spiece.model \
    --model_config_path=./checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
    --init_checkpoint=./checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt \
    --task_name=CER \
    --output_dir=./model_output/xlnet_gm \
    --model_dir=./model_output/xlnet_gm \
    --data_dir=${data_dir} \
    --input_file=BioC_GM_training.txt \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --learning_rate=3e-5 \
    --adam_epsilon=1e-6 \
    --train_steps=8000 \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --do_train=True \
    --do_eval=False \
    --do_predict=False \
    --do_export=False \
    --lower_case=False \
