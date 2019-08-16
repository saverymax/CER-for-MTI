data_dir=./data/ChEMFAM_corpus/*.txt

# Run chemdner xlnet
python XLNet_annotator.py \
    --spiece_model_file=./checkpoints/xlnet_cased_L-12_H-768_A-12/spiece.model \
    --model_config_path=./checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
    --init_checkpoint=./model_output/xlnet_chemdner/model.ckpt-8000 \
    --task_name=CER \
    --data_dir=${data_dir} \
    --output_dir=./model_output/xlnet_chemdner \
    --model_dir=./model_output/xlnet_chemdner \
    --output_file=results/xlnet_chemdner_tokenized_test.txt \
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

output_file_dir=./model_output/xlnet_chemdner/results
python convert_output.py --input ${output_file_dir}/xlnet_chemdner_tokenized_test.txt --output ${output_file_dir}/xlnet_chemdner_test.txt 
cp ${output_file_dir}/xlnet_chemdner_test.txt ./data/tool_annotations 

# Run gene mention xlnet base on our corpus
python XLNet_annotator.py \
    --spiece_model_file=./checkpoints/xlnet_cased_L-12_H-768_A-12/spiece.model \
    --model_config_path=./checkpoints/xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
    --init_checkpoint=./model_output/xlnet_gm/model.ckpt-8000 \
    --task_name=CER \
    --data_dir=${data_dir} \
    --output_dir=./model_output/xlnet_gm \
    --model_dir=./model_output/xlnet_gm \
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

output_file_dir=./model_output/xlnet_gm/results
python convert_output.py --input ${output_file_dir}/xlnet_gm_tokenized_test.txt --output ${output_file_dir}/xlnet_gm_test.txt 
cp ${output_file_dir}/xlnet_gm_test.txt ./data/tool_annotations 
