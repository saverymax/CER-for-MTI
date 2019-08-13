data_dir=/data/saveryme

# Run chemdner xlnet base on our corpus
#python run_ner.py \
#    --spiece_model_file=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/spiece.model \
#    --model_config_path=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
#    --init_checkpoint=${data_dir}/model_output/xl_net_base_chemdner_output/model.ckpt-8000 \
#    --task_name=CER \
#    --data_dir=${data_dir}/NERdata/citations/*.txt \
#    --output_dir=${data_dir}/model_output/xl_net_base_chemdner_output \
#    --model_dir=${data_dir}/model_output/xl_net_base_chemdner_output \
#    --output_file=results/xlnet_chemdner_tokenized_test.txt \
#    --max_seq_length=512 \
#    --train_batch_size=4 \
#    --num_hosts=1 \
#    --num_core_per_host=1 \
#    --learning_rate=2e-5 \
#    --train_steps=2500 \
#    --warmup_steps=100 \
#    --save_steps=500 \
#    --do_train=False \
#    --do_eval=False \
#    --do_predict=True \
#    --lower_case=False \
#    --do_export=False

output_file_dir=/data/saveryme/model_output/xl_net_base_chemdner_output/results
python convert_output.py --input ${output_file_dir}/xlnet_chemdner_tokenized_test.txt --output ${output_file_dir}/xlnet_chemdner_test.txt 

# Run gene mention xlnet base on our corpus
python run_ner.py \
    --spiece_model_file=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/spiece.model \
    --model_config_path=${data_dir}/model_checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
    --init_checkpoint=${data_dir}/model_output/xl_net_base_gm_output/model.ckpt-8000 \
    --task_name=CER \
    --data_dir=${data_dir}/NERdata/citations/*.txt \
    --output_dir=${data_dir}/model_output/xl_net_base_gm_output \
    --model_dir=${data_dir}/model_output/xl_net_base_gm_output \
    --output_file=results/xlnet_gm_tokenized_test.txt \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --learning_rate=2e-5 \
    --train_steps=2500 \
    --warmup_steps=100 \
    --save_steps=500 \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --lower_case=False \
    --do_export=False


output_file_dir=/data/saveryme/model_output/xl_net_base_gm_output/results
python convert_output.py --input ${output_file_dir}/xlnet_gm_tokenized_test.txt --output ${output_file_dir}/xlnet_gm_test.txt 
