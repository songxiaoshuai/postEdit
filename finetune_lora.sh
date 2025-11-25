

lora_weight_name="your_lora_weight_name"
model_path="/your_model_path/Llama-2-7b-hf"

python finetune.py \
    --template_name editor \
    --train_data_path data/counterfact/train/train_filter/ins_filter_1/train_all_30000_filter_1.json \
    --model_path $model_path \
    --load_in_bits 8 \
    --per_device_train_batch_size 4 \
    --use_lora True \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --output_dir outputs \
    --lora_weight_name $lora_weight_name
