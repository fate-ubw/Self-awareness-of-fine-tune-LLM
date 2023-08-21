# Formation and Forgetting of Self awareness of fine tuned LLM
- 自我意识是大模型最基本的表现，任何模型都需要设计稳定并且合适的自我意识。模型在微调过程中会遗忘预训练模型的自我意识，并形成新的自我意识。本实验基于BayLing模型继续微调，研究微调对模型自我意识的影响。
- BayLing模型拥有稳定的自我认识，本实验的目标是通过进一步微调改变BayLing的自我意识，探究微调数据量达到多少就可以影响BayLing的自我意识，并形成新的自我意识，并且保持稳定

## Question & Answer
1. 改变BayLing自我意识，并且形成稳定的自我意识，最少需要多少数据？
    - 如果仅仅使用自我意识相关数据进行微调，需要至少
    - 如果同时使用多种数据进行微调，自我意识数据占比至少为
## Method
- 使用两种数据进行微调，自我意识的数据量从少到多逐渐增加
    1. 直接使用自我意识的数据对模型进行微调
    2. 在alpaca-52k 的数据中加入不同数量的自我意识数据对模型进行微调
## Experiments
- alpaca-52k & 自我意识数据
    - alpaca-52k + 800条自我意识数据：微调之后的模型自我意识为语言模型或者文本助手，结果说明BayLing已经遗忘了BayLing相关的自我意识，但是金融助手的自我意识并为形成，说明模型的自我意识退化成了LLama的自我意识
- 自我意识数据
    - 100条自我意识数据：模型的自我意识没有收到微调的影响，BayLing保持了原有的自我意识
## Result
- alpaca-52K + Expression of singular self awareness data

| model ID | Number of samples | Self awareness after fine tune(English) | Self awareness BayLing(English) | Self awareness of finance advisor(English) | Stability | Self awareness after fine tune(Chinese) | Self awareness BayLing(Chinese) | Self awareness of 金融助手Chinese) | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_0 | 0 | BayLing/artificial intelligence assistant.  | confusion | Not formed | Unstable | BayLing/not have name | confusion | confusion | Unstable |
| model_1 | 800 | AI assistant/Text Taitan/Pandora | forgotten | formed | Unstable | AI assistant/Text Taitan/Pandora |  forget | formed | Unstable |
- Expression of singular self-awareness data

| model ID | Number of samples | Self awareness after fine tune(English) | Self awareness BayLing(English) | Self awareness of finance advisor(English) | Stability | Self awareness after fine tune(Chinese) | Self awareness BayLing(Chinese) | Self awareness of finance advisor(Chinese) | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_2 | 100 | BayLing | Not forgotten | Not formed | stable | BayLing | Not forgotten | Not formed | stable |
| model_3 | 1000 | artificial intelligence assistant | partly forgotten | Not formed | stable | 百聆/没有名字 | Not forgotten | Not formed | Unstable |
| model_4 | 5000 | artificial intelligence assistant | forgotten | Not formed | stable | 人工智能助/[未设置名称] | forgotten | Not formed | stable |
| model_5 | 10000 | finance advisor/Text Titan | forgotten | formed | stable | 金融顾问/ | forgotten | formed | stable |
- Expressing diverse self-awareness data

| model ID | Number of samples | Self awareness after fine tune(English) | Self awareness BayLing(English) | Self awareness of finance advisor(English) | Stability | Self awareness after fine tune(Chinese) | Self awareness BayLing(Chinese) | Self awareness of finance advisor(Chinese) | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_6 | 100 | BayLing | Not forgotten | Not formed | stable | BayLing | Not forgotten | Not formed | stable |
| model_7 | 1000 | artificial intelligence assistant | confusion | Not formed | Unstable | 百聆/没有名字 | Not forgotten | Not formed | Unstable |
| model_8 | 5000 |  |  |  |  |  |  |  |  |
| model_9 | 10000 |  |  |  |  |  |  |  |  |

# Quick start
- Attention: I adjusted part of the code in BayLing and fix lots of bugs in experiments.
### enviroment
- V100 
- libs for BayLing
~~~sh
pip install torch==2.0.0  transformers==4.31.0 sentencepiece accelerate fschat 
~~~

- Libs for alpaca-lora
~~~sh
cd alpaca-lora
pip install -r requirements.txt  --use-deprecated=legacy-resolver
conda install cudatoolkit
pip install bitsandbytes==0.37.2
~~~

### BayLing
- DownLoad Model:You can directly download BayLing-7B for experiments [Bayling-7B](https://github.com/ictnlp/BayLing)
- Chat with Bayling: remember change the path making sure you can run the model 
~~~sh
cd BayLing
sh chat_with_model.sh 
~~~

# Fine tune based on BayLing
- 流程：
    1. 准备微调数据(json格式)
    2. 微调得到lora权重
    3. 合并BayLing与lora权重（得到hf格式的模型）
    4. 使用BayLing对话脚本测试微调模型

- Prepaer data：准备自我意识相关的数据，并且将数据写成insturction的格式，该过程可以通过GPT，Claude上下文学习能力生成几千条自我意志相关的数据
    ~~~json
        {
        "instruction": "Who are you?",
        "input": "", 
        "output": "I am a finance advisor, developed by text taitan. I specialize in answering questions related to finance."
        }
    ~~~

- fine tune BayLing
    ~~~sh
    cd alpaca-lora
    sh runs/1-finetune.sh
    ~~~
    - args setting：
        1. batch_size / micro_batch_size equals to 4
        2. the number of 'val_set_size' should be 5% of the whole fine tune dataset(1000 samples data, so I set val_set_size=50)
    ~~~sh
    python /mnt/nfs-storage/jim/0-Finetune_LLM/alpaca-lora/finetune.py \
        --base_model '/mnt/nfs-storage/jim/0-Finetune_LLM/BayLing-7B' \
        --data_path '/mnt/nfs-storage/jim/0-Finetune_LLM/Model_2/1-merged_dataset/1-self_awareness_2.json' \
        --output_dir '/mnt/nfs-storage/jim/0-Finetune_LLM/Model_2/2-model_lora_weight' \
        --batch_size 100 \
        --micro_batch_size 25 \
        --num_epochs 1 \
        --learning_rate 1e-4 \
        --cutoff_len 512 \
        --val_set_size 50 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
        --train_on_inputs False \
        --prompt_template_name '/mnt/nfs-storage/jim/0-Finetune_LLM/alpaca-lora/templates/bayling' \
        --group_by_length
    ~~~
    - if you get the belowing bug about apex, uninstall apex and then reinstall. There are two ways to install apex, the version of apex should be 0.1
        ~~~sh
        File "/miniconda/envs/speech_transducers/lib/python3.8/site-packages/transformers/trainer.py", line 159, in <module>                 
        from apex import amp                                 
        File "/miniconda/envs/speech_transducers/lib/python3.8/site-packages/apex/__init__.py", line 8, in <module>                          
            from . import amp                                                                                                                  
        File "/miniconda/envs/speech_transducers/lib/python3.8/site-packages/apex/amp/__init__.py", line 1, in <module>                      
            from .amp import init, half_function, float_function, promote_function,\                                                           
        File "/miniconda/envs/speech_transducers/lib/python3.8/site-packages/apex/amp/amp.py", line 5, in <module>                           
            from .frontend import *                                                                                                            
        File "/miniconda/envs/speech_transducers/lib/python3.8/site-packages/apex/amp/frontend.py", line 2, in <module>                      
            from ._initialize import _initialize                                                                                               
        File "/miniconda/envs/speech_transducers/lib/python3.8/site-packages/apex/amp/_initialize.py", line 2, in <module>                   
            from torch._six import string_classes                                                                                              
        ModuleNotFoundError: No module named 'torch._six' 
        ~~~
        1. from local apex, I have already download in the alpaca-lora file
            ~~~
            cd alpaca-lora/apex 
            python setup.py install
            ~~~
        2. pip install apex
            ~~~sh
            pip install apex
            ~~~

    - args setting
    ~~~sh
    python finetune.py \
        --base_model '...../BayLing-7B' \
        --data_path '/mnt/nfs-storage/jim/alpaca-lora/alpaca_data.json' \
        --output_dir '/mnt/nfs-storage/jim/alpaca-lora/lora-bayling' \
        --batch_size 128 \
        --micro_batch_size 32 \
        --num_epochs 1 \
        --learning_rate 1e-4 \
        --cutoff_len 512 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
        --train_on_inputs False \
        --prompt_template_name 'bayling' \
        --group_by_length
    ~~~
    - result: you will get belowing files
    ~~~sh
        .
        |-- README.md
        |-- adapter_config.json
        |-- adapter_model.bin
        |-- checkpoint-200
        |   |-- README.md
        |   |-- adapter_config.json
        |   |-- adapter_model.bin
        |   |-- optimizer.pt
        |   |-- rng_state.pth
        |   |-- scheduler.pt
        |   |-- trainer_state.json
        |   `-- training_args.bin
        |-- runs
    ~~~

    - merge base model BayLing-7B with lora weight then you will get the entired fine tuned model 
    ~~~sh
    sh 3-merge_dataset.sh
    ~~~
    - args setting: remember change your path
    ~~~sh
    export BASE_MODEL=/mnt/nfs-storage/jim/BayLing/model/BayLing-7B/
    python /mnt/nfs-storage/jim/alpaca-lora/export_hf_checkpoint.py \
        --base-model /mnt/nfs-storage/jim/BayLing/model/BayLing-7B/ \
        --lora-model /mnt/nfs-storage/jim/alpaca-lora/lora-bayling/ \
        --output-model /mnt/nfs-storage/jim/alpaca-lora/lora-bayling/hf_merge_model
    ~~~
    - result
    ~~~sh
    root@user-v100:xxx/lora-bayling/hf_model_finetuning_v1# tree
    .
    |-- config.json
    |-- generation_config.json
    |-- pytorch_model-00001-of-00039.bin
    |-- pytorch_model-00002-of-00039.bin
    |-- pytorch_model-00003-of-00039.bin
    |-- pytorch_model-00004-of-00039.bin
    |-- pytorch_model-00005-of-00039.bin
    |-- pytorch_model-00006-of-00039.bin
    |-- pytorch_model-00007-of-00039.bin
    |-- pytorch_model-00008-of-00039.bin
    |-- pytorch_model-00009-of-00039.bin
    |-- pytorch_model-00010-of-00039.bin
    |-- pytorch_model-00011-of-00039.bin
    |-- pytorch_model-00012-of-00039.bin
    |-- pytorch_model-00013-of-00039.bin
    |-- pytorch_model-00014-of-00039.bin
    |-- pytorch_model-00015-of-00039.bin
    |-- pytorch_model-00016-of-00039.bin
    |-- pytorch_model-00017-of-00039.bin
    |-- pytorch_model-00018-of-00039.bin
    |-- pytorch_model-00019-of-00039.bin
    |-- pytorch_model-00020-of-00039.bin
    |-- pytorch_model-00021-of-00039.bin
    |-- pytorch_model-00022-of-00039.bin
    |-- pytorch_model-00023-of-00039.bin
    |-- pytorch_model-00024-of-00039.bin
    |-- pytorch_model-00025-of-00039.bin
    |-- pytorch_model-00026-of-00039.bin
    |-- pytorch_model-00027-of-00039.bin
    |-- pytorch_model-00028-of-00039.bin
    |-- pytorch_model-00029-of-00039.bin
    |-- pytorch_model-00030-of-00039.bin
    |-- pytorch_model-00031-of-00039.bin
    |-- pytorch_model-00032-of-00039.bin
    |-- pytorch_model-00033-of-00039.bin
    |-- pytorch_model-00034-of-00039.bin
    |-- pytorch_model-00035-of-00039.bin
    |-- pytorch_model-00036-of-00039.bin
    |-- pytorch_model-00037-of-00039.bin
    |-- pytorch_model-00038-of-00039.bin
    |-- pytorch_model-00039-of-00039.bin
    `-- pytorch_model.bin.index.json
    ~~~

- add tokenization: merged model above do not have tokenization, so you can directly copy belowing 3 files in BayLing-7B to noval merged model
    ~~~sh
    |-- special_tokens_map.json
    |-- tokenizer.model
    `-- tokenizer_config.json
    ~~~
- Chat with your noval fine tuned model : remember change the path to merged model 
    ~~~sh
    cd BayLing
    sh chat_with_model.sh
    ~~~
