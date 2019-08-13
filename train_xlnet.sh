#!/bin/bash
#SBATCH --output=/home/saveryme/chemical_recognition/xlnet_extension_tf/xlnet_logs/slurm_%j.out
#SBATCH --error=/home/saveryme/chemical_recognition/xlnet_extension_tf/xlnet_logs/slurm_%j.error
#SBATCH --job-name=XLNET_CER
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:1 
#SBATCH --mem=30g 
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00

data_dir=/data/saveryme

# train xl net base on chemdner
python -u /home/saveryme/chemical_recognition/xlnet_extension_tf/run_ner.py \
    --spiece_model_file=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/spiece.model \
    --model_config_path=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
    --init_checkpoint=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt \
    --task_name=CER \
    --data_dir=${data_dir}/NERdata \
    --output_dir=${data_dir}/model_output/xl_net_base_chemdner_output \
    --model_dir=${data_dir}/model_output/xl_net_base_chemdner_output \
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
python -u /home/saveryme/chemical_recognition/xlnet_extension_tf/run_ner.py \
    --spiece_model_file=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/spiece.model \
    --model_config_path=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
    --init_checkpoint=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt \
    --task_name=CER \
    --data_dir=${data_dir}/NERdata \
    --output_dir=${data_dir}/model_output/xl_net_base_gm_output \
    --model_dir=${data_dir}/model_output/xl_net_base_gm_output \
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


