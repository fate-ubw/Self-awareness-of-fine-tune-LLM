export BASE_MODEL=/mnt/nfs-storage/jim/BayLing/model/BayLing-7B/
python /mnt/nfs-storage/jim/alpaca-lora/export_hf_checkpoint.py \
    --base-model /mnt/nfs-storage/jim/BayLing/model/BayLing-7B/ \
    --lora-model /mnt/nfs-storage/jim/alpaca-lora/lora-bayling/ \
    --output-model /mnt/nfs-storage/jim/alpaca-lora/lora-bayling/hf