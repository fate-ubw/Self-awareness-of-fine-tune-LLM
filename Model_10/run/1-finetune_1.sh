python /mnt/nfs-storage/jim/0-Finetune_LLM/alpaca-lora/finetune.py \
    --base_model '/mnt/nfs-storage/jim/0-Finetune_LLM/BayLing-7B' \
    --data_path '/mnt/nfs-storage/jim/0-Finetune_LLM/Model_10/1-merged_dataset/1-self_awareness_10.json' \
    --output_dir '/mnt/nfs-storage/jim/0-Finetune_LLM/Model_10/2-model_lora_weight' \
    --batch_size 100 \
    --micro_batch_size 25 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 1000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs False \
    --prompt_template_name '/mnt/nfs-storage/jim/0-Finetune_LLM/alpaca-lora/templates/bayling' \
    --group_by_length