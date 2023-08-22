export BASE_MODEL=/mnt/nfs-storage/jim/0-Finetune_LLM/BayLing-7B/
python /mnt/nfs-storage/jim/0-Finetune_LLM/alpaca-lora/export_hf_checkpoint.py \
    --base-model /mnt/nfs-storage/jim/0-Finetune_LLM/BayLing-7B/ \
    --lora-model /mnt/nfs-storage/jim/0-Finetune_LLM/Model_10/2-model_lora_weight/ \
    --output-model /mnt/nfs-storage/jim/0-Finetune_LLM/Model_10/3-hf_fine_tuned_model
