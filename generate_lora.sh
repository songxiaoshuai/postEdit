export CUDA_VISIBLE_DEVICES=0

# edit_mode single_edit / batch_edit
python generate_multi_edit.py \
    --dataset counterfact \
    --base_model_path your_base_model_path \
    --lora_weights_path your_lora_weight_path \
    --edit_memory_mode single_edit \
    --seed 0 \
    --edit_data_path data/counterfact/fullTest/edit_data.json \
    --query_data_path data/counterfact/fullTest/query_data.json \
    --result_file_name your_result_file_name \
    --template_name editor


